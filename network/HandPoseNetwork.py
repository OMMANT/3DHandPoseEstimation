from __future__ import print_function, unicode_literals
import os, pickle

from utils.general import *
from utils.utils import multiply
import matplotlib.pyplot as plt

class ColorHandPose3DNetwork(object):
    def __init__(self, hand_side='left'):
        self.crop_size = 256
        self.J = 21

        self.setup_HandSegNet()
        self.setup_PoseNet2D()
        self.setup_PosePrior()
        self.setup_GestureNet()
        hand_side = tf.constant([[0., 1.]]) if hand_side == 'right' else tf.constant([[1., 0.]])
        self.hand_side = hand_side

    def inference(self, image):
        """ Full pipeline: HandSegNet + PoseNet + PosePrior.

            Inputs:
                image: [B, H, W, 3] tf.float32 tensor, Image with mean subtracted
                hand_side: [B, 2] tf.float32 tensor, One hot encoding if the image is showing left or right side
                evaluation: [] tf.bool tensor, True while evaluation false during training (controls dropout)

            Outputs:
                hand_scoremap: [B, 256, 256, 2] tf.float32 tensor, Scores for background and hand class
                image_crop: [B, 256, 256, 3] tf.float32 tensor, Hand cropped input image
                scale_crop: [B, 1] tf.float32 tensor, Scaling between input image and image_crop
                center: [B, 1] tf.float32 tensor, Center of image_crop wrt to image
                keypoints_scoremap: [B, 256, 256, 21] tf.float32 tensor, Scores for the hand keypoints
                keypoint_coord3d: [B, 21, 3] tf.float32 tensor, Normalized 3D coordinates
        """
        hand_scoremap = self.inference_detection(image)
        hand_scoremap = hand_scoremap[-1]

        # Intermediate data processing
        hand_mask = single_obj_scoremap(hand_scoremap)
        center, best_crop_size = calc_center_bb(hand_mask)
        best_crop_size *= 1.25
        scale_crop = tf.minimum(tf.maximum(self.crop_size / best_crop_size, .25), 5.)
        image_crop = crop_image_from_xy(image, center, self.crop_size, scale_crop)

        #detect keypoints in 2D
        keypoints_scoremap = self.inference_pose2d(image_crop)
        keypoints_scoremap = keypoints_scoremap[-1]

        # estimate most likely 3D pose
        keypoint_coord3d = self.inference_pose3d(keypoints_scoremap)

        # upsample keypoint scoremap
        s = image_crop.shape
        keypoints_scoremap = tf.image.resize(keypoints_scoremap, (s[1], s[2]))

        return [hand_scoremap, image_crop, scale_crop, center, keypoints_scoremap, keypoint_coord3d]


    def inference_detection(self, image):
        """ Only 2D part of the pipeline: HandSegNet + PoseNet.

            Inputs:
                image: [B, H, W, 3] tf.float32 tensor, Image with mean subtracted

            Outputs:
                image_crop: [B, 256, 256, 3] tf.float32 tensor, Hand cropped input image
                scale_crop: [B, 1] tf.float32 tensor, Scaling between input image and image_crop
                center: [B, 1] tf.float32 tensor, Center of image_crop wrt to image
                keypoints_scoremap: [B, 256, 256, 21] tf.float32 tensor, Scores for the hand keypoints
        """
        scoremap_list = []
        x = image
        scoremap = self.HandSegNet.predict(x)
        scoremap_list.append(scoremap)

        return [tf.image.resize(x, (256, 256)) for x in scoremap_list]

    def inference_pose2d(self, image_crop):
        name_list = ['conv5_2', 'conv6_7', 'conv7_7']
        outputs = [self.PoseNet2D.get_layer(name=name).output for name in name_list]
        functions = [tf.keras.backend.function([self.PoseNet2D.input], [output]) for output in outputs]
        score_map_list = [np.array(func([image_crop])[0]) for func in functions]
        return score_map_list

    def inference_pose3d(self, keypoints_scoremap):
        # infer coordinates in the canonical frame
        coord_can = self.inference_pose3d_can(keypoints_scoremap) # (B, 21, 3)

        # infer viewpoint
        rot_mat = self.inference_viewpoint(keypoints_scoremap)

        # flip hand according to hand side
        cond_right = tf.equal(tf.argmax(self.hand_side, 1), 1)
        cond_right_all = tf.tile(tf.reshape(cond_right, [-1, 1, 1]), [1, self.J, 3]) # (1, 21, 3)
        coord_xyz_can_flip = self.flip_right_hand(coord_can, cond_right_all)

        # rotate view back
        coord_xyz_rel_normed = tf.matmul(coord_xyz_can_flip, rot_mat)

        return coord_xyz_rel_normed



    def inference_pose3d_can(self, keypoints_scoremap):
        coord_xyz_rel = [self.PosePrior.predict(np.expand_dims(x, axis=0)) for x in keypoints_scoremap]
        coord_xyz_rel = tf.reshape(coord_xyz_rel, [keypoints_scoremap.shape[0], self.J, 3])
        return coord_xyz_rel

    def inference_viewpoint(self, keypoints_scoremap):
        ux, uy, uz = self.rotation_estimation(keypoints_scoremap)

        rot_mat = self.get_rot_mat(ux, uy, uz)

        return rot_mat


    def rotation_estimation(self, scoremap):
        scoremap = tf.concat([scoremap], 3)
        output = np.array([self.GestureNet.predict(np.expand_dims(x, axis=0)) for x in scoremap]) # (B, 3, 1, 1)

        return output[:, 0, 0, :], output[:, 1, 0, :], output[:, 2, 0, :]


    def get_rot_mat(self, ux_b, uy_b, uz_b):
        theta = tf.math.sqrt(tf.square(ux_b) + tf.math.square(uy_b) + tf.math.square(uz_b) + 1e-8)

        st = tf.sin(theta)[:, 0]
        ct = tf.cos(theta)[:, 0]
        one_ct = (1 - tf.cos(theta))[:, 0]
        norm_fac = 1. / theta[:, 0]
        ux = ux_b[:, 0] * norm_fac
        uy = uy_b[:, 0] * norm_fac
        uz = uz_b[:, 0] * norm_fac

        trafo_matrix = self.stitch_mat_from_vecs(
            [ct + ux * ux * one_ct, ux * uy * one_ct - uz * st, ux * uz * one_ct + uy * st,
             uy * ux * one_ct + uz * st, ct + uy * uy * one_ct, uy * uz * one_ct - ux * st,
             uz * ux * one_ct - uy * st, uz * uy * one_ct + ux * st, ct + uz * uz * one_ct])

        return trafo_matrix

    def stitch_mat_from_vecs(self, vector):
        batch_size = vector[0].shape[0]
        vector = [tf.reshape(x, [1, batch_size]) for x in vector]

        trafo_matrix = tf.dynamic_stitch([[0], [1], [2],
                                          [3], [4], [5],
                                          [6], [7], [8]], vector)
        trafo_matrix = tf.reshape(trafo_matrix, [3, 3, batch_size])
        trafo_matrix = tf.transpose(trafo_matrix, [2, 0, 1])

        return trafo_matrix

    def flip_right_hand(self, coords_xyz_canonical, cond_right):
        expanded = False
        s = coords_xyz_canonical.shape
        # if batch_size is 1
        if len(s) == 2:
            coords_xyz_canonical = tf.expand_dims(coords_xyz_canonical, axis=0)
            cond_right = tf.expand_dims(cond_right, 0)
            expanded = True

        temp = tf.stack([coords_xyz_canonical[:, :, 0], coords_xyz_canonical[:, :, 1], coords_xyz_canonical[:, :, 2]], -1)
        coords_xyz_left = tf.where(cond_right, temp, coords_xyz_canonical)

        if expanded:
            coords_xyz_left = tf.squeeze(coords_xyz_left, [0])
        return coords_xyz_left


    def setup_HandSegNet(self):
        # Setup HandSegNet
        self.HandSegNet = tf.keras.Sequential([
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=64, name='conv1_1',
                                   input_shape=(240, 320, 3)),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=64, name='conv1_2'),
            tf.keras.layers.MaxPool2D(name='pool1'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=128, name='conv2_1'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=128, name='conv2_2'),
            tf.keras.layers.MaxPool2D(name='pool2'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=256, name='conv3_1'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=256, name='conv3_2'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=256, name='conv3_3'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=256, name='conv3_4'),
            tf.keras.layers.MaxPool2D(name='pool3'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=512, name='conv4_1'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=512, name='conv4_2'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=512, name='conv4_3'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=512, name='conv4_4'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=512, name='conv5_1'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=128, name='conv5_2'),
            tf.keras.layers.Conv2D(kernel_size=(1, 1), activation='relu', padding='same', filters=512, name='conv6_1'),
            tf.keras.layers.Conv2D(kernel_size=(1, 1), activation='relu', padding='same', filters=2, name='conv6_2'),
        ])
        with open('./res/weights/handsegnet-rhd.pickle', 'rb') as file:
            weight_dict = pickle.load(file)
        # Setup params to pretrained model
        for layer in self.HandSegNet.layers:
            if layer.name.startswith('pool'):
                continue
            weight = layer.weights
            name = 'HandSegNet/' + layer.name + '/'
            weight[0].assign(weight_dict[name + 'weights'])
            weight[1].assign(weight_dict[name + 'biases'])

    def setup_PoseNet2D(self):
        self.PoseNet2D = tf.keras.Sequential([
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=64, name='conv1_1',
                                   input_shape=(256, 256, 3)),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=64, name='conv1_2'),
            tf.keras.layers.MaxPool2D(name='pool1'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=128, name='conv2_1'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=128, name='conv2_2'),
            tf.keras.layers.MaxPool2D(name='pool2'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=256, name='conv3_1'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=256, name='conv3_2'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=256, name='conv3_3'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=256, name='conv3_4'),
            tf.keras.layers.MaxPool2D(name='pool3'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=512, name='conv4_1'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=512, name='conv4_2'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=256, name='conv4_3'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=256, name='conv4_4'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=256, name='conv4_5'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=256, name='conv4_6'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=128, name='conv4_7'),
            tf.keras.layers.Conv2D(kernel_size=(1, 1), activation='relu', padding='same', filters=512, name='conv5_1'),
            tf.keras.layers.Conv2D(kernel_size=(1, 1), activation='relu', padding='same', filters=self.J,
                                   name='conv5_2'),
        ])
        encoding = self.PoseNet2D.get_layer(name='conv4_7').output
        x = self.PoseNet2D.get_layer(name='conv5_2').output
        iter_num = 2
        layer_num = 7
        for i in range(iter_num):
            concat_layer = tf.keras.layers.Concatenate(name='concat{}'.format(i + 1))([encoding, x])
            self.PoseNet2D = tf.keras.Model(inputs=self.PoseNet2D.input, outputs=concat_layer)
            x = concat_layer
            for j in range(1, layer_num + 1):
                k_size = 7 if j < 6 else 1
                filters = 128 if j < 7 else self.J
                name = 'conv{}_{}'.format(i + 6, j)
                x = tf.keras.layers.Conv2D(kernel_size=(k_size, k_size), activation='relu' if j < 7 else None,
                                           padding='same', filters=filters, name=name)(x)
            self.PoseNet2D = tf.keras.Model(inputs=self.PoseNet2D.input, outputs=x)
        # Setup params with pre-trained value.
        with open('./res/weights/posenet3d-rhd-stb-slr-finetuned.pickle', 'rb') as file:
            weight_dict = pickle.load(file)
            valid_layer_name = [key.split('/')[1] for key in weight_dict.keys()]

        for layer in self.PoseNet2D.layers:
            weight = layer.weights
            if layer.name not in valid_layer_name:
                continue
            name = 'PoseNet2D/' + layer.name + '/'
            weight[0].assign(np.array(weight_dict[name + 'weights']))
            weight[1].assign(weight_dict[name + 'biases'])

    def setup_PosePrior(self):
        self.PosePrior = tf.keras.Sequential([
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=32,
                                   name='conv_pose_0_1', input_shape=(32, 32, 21)),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', filters=32,
                                   name='conv_pose_0_2'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=64,
                                   name='conv_pose_1_1',
                                   input_shape=(32, 32, 21)),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', filters=64,
                                   name='conv_pose_1_2'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=128,
                                   name='conv_pose_2_1',
                                   input_shape=(32, 32, 21)),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', filters=128,
                                   name='conv_pose_2_2'),
        ])
        x = self.PosePrior.layers[-1].output
        temp = tf.keras.layers.Flatten()(x)
        x = tf.concat([temp, tf.constant([[1., 0.]])], 1)
        for i, units in enumerate([512, 512]):
            x = tf.keras.layers.Dense(512, activation='relu', name='fc_rel%d' % i)(x)
            x = tf.keras.layers.Dropout(.2)(x)
        x = tf.keras.layers.Dense(self.J * 3, name='fc_xyz')(x)
        self.PosePrior = tf.keras.Model(inputs=self.PosePrior.inputs, outputs=x)
        self.PosePrior.trainable = False

        # Setup params with pre-trained value.
        with open('./res/weights/posenet3d-rhd-stb-slr-finetuned.pickle', 'rb') as file:
            weight_dict = pickle.load(file)
            valid_layer_name = [key.split('/')[1] for key in weight_dict.keys()]

        for layer in self.PosePrior.layers:
            weight = layer.weights
            if layer.name not in valid_layer_name:
                continue
            name = 'PosePrior/' + layer.name + '/'
            weight[0].assign(np.array(weight_dict[name + 'weights']))
            weight[1].assign(weight_dict[name + 'biases'])

    def setup_GestureNet(self, hand_side=tf.constant([[1., 0.]])):
        self.GestureNet = tf.keras.Sequential([
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=64,
                                   name='conv_vp_0_1', input_shape=(32, 32, 21)),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', filters=64,
                                   name='conv_vp_0_2'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=128,
                                   name='conv_vp_1_1'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', filters=128,
                                   name='conv_vp_1_2'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), activation='relu', padding='same', filters=256,
                                   name='conv_vp_2_1'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', filters=256,
                                   name='conv_vp_2_2'),
        ])
        x = self.GestureNet.layers[-1].output
        x = tf.keras.layers.Flatten()(x)
        x = tf.concat([x, hand_side], 1)
        x = tf.keras.layers.Dense(256, activation='relu', name='fc_vp0')(x)
        x = tf.keras.layers.Dropout(.25)(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='fc_vp1')(x)
        x = tf.keras.layers.Dropout(.25)(x)
        ux = tf.keras.layers.Dense(1, activation='relu', name='fc_vp_ux')(x)
        uy = tf.keras.layers.Dense(1, activation='relu', name='fc_vp_uy')(x)
        uz = tf.keras.layers.Dense(1, activation='relu', name='fc_vp_uz')(x)
        self.GestureNet = tf.keras.Model(inputs=self.GestureNet.input, outputs=[ux, uy, uz])

        # Setup params with pre-trained value
        with open('./res/weights/posenet3d-rhd-stb-slr-finetuned.pickle', 'rb') as file:
            weight_dict = pickle.load(file)
            valid_layer_name = [key.split('/')[1] for key in weight_dict.keys()]

        for layer in self.GestureNet.layers:
            weight = layer.weights
            if layer.name not in valid_layer_name:
                continue
            name = 'ViewpointNet/' + layer.name + '/'
            weight[0].assign(np.array(weight_dict[name + 'weights']))
            weight[1].assign(weight_dict[name + 'biases'])

    def train_HandSegNet(self, train_X, train_Y, epochs=10, batch_size=8,
                         optimizers=tf.keras.optimizers.Adam(), loss='mse', val_data=None):
        self.HandSegNet.compile(optimizer=optimizers,
                                loss=loss,
                                metrics=['accuracy'])
        self.HandSegNet.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, validation_data=val_data)


