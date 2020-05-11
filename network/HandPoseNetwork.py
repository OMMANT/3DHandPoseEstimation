from __future__ import print_function, unicode_literals
import os, pickle

from utils.general import *
import matplotlib.pyplot as plt
from .Layers import *

class ColorHandPose3DNetwork(object):
    def __init__(self, hand_side='left'):
        self.crop_size = 256
        self.J = 21
        hand_side = tf.constant([[0., 1.]]) if hand_side == 'right' else tf.constant([[1., 0.]])
        self.hand_side = hand_side

        self.setup_HandSegNet()
        self.setup_PoseNet2D()
        self.setup_PosePrior()
        self.setup_GestureNet()

    def inference(self, image):
        """ Full pipeline: HandSegNet + PoseNet + PosePrior.
            Inputs:
                image: [B, 240, 320, 3] tf.float32 tensor, Image with mean subtracted
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
        """ HandSegNet: Detects the hand in the input image by segmenting it.
            Inputs:
                image: [B, H, W, 3] tf.float32 tensor, Image with mean subtracted
            Outputs:
                scoremap_list_large: list of [B, 256, 256, 2] tf.float32 tensor, Scores for the hand segmentation classes
        """
        scoremap_list = []
        x = image
        scoremap = self.HandSegNet.predict(x)
        scoremap_list.append(scoremap)
        shape_of_img = image.shape
        return [tf.image.resize(x, (shape_of_img[1], shape_of_img[2])) for x in scoremap_list]

    def inference_pose2d(self, image_crop):
        """ PoseNet: Given an image it detects the 2D hand keypoints.
            The image should already contain a rather tightly cropped hand.
            Inputs:
                image: [B, 240, 320, 3] tf.float32 tensor, Image with mean subtracted
            Outputs:
                scoremap_list_large: list of [B, 256, 256, 21] tf.float32 tensor, Scores for the hand keypoints
        """
        name_list = ['conv5_2', 'conv6_7', 'conv7_7']
        outputs = [self.PoseNet2D.get_layer(name=name).output for name in name_list]
        functions = [tf.keras.backend.function([self.PoseNet2D.input], [output]) for output in outputs]
        score_map_list = [np.array(func([image_crop])[0]) for func in functions]
        return score_map_list

    def inference_pose3d(self, keypoints_scoremap):
        """ PosePrior + Viewpoint: Estimates the most likely normalized 3D pose given 2D detections and hand side.
            Inputs:
                keypoints_scoremap: [B, 32, 32, 21] tf.float32 tensor, Scores for the hand keypoints
            Outputs:
                coord_xyz_rel_normed: [B, 21, 3] tf.float32 tensor, Normalized 3D coordinates
        """
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
        """ Inference of canonical coordinates. """
        ux, uy, uz = self.rotation_estimation(keypoints_scoremap)

        rot_mat = self.get_rot_mat(ux, uy, uz)

        return rot_mat

    def rotation_estimation(self, scoremap):
        """ Estimates the rotation from canonical coords to realworld xyz. """
        scoremap = tf.concat([scoremap], 3)
        output = np.array([self.GestureNet.predict(np.expand_dims(x, axis=0)) for x in scoremap]) # (B, 3, 1, 1)

        return output[:, 0, 0, :], output[:, 1, 0, :], output[:, 2, 0, :]

    def get_rot_mat(self, ux_b, uy_b, uz_b):
        """ Returns a rotation matrix from axis and (encoded) angle."""
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
        """ Stitches a given list of vectors into a 3x3 matrix."""
        batch_size = vector[0].shape[0]
        vector = [tf.reshape(x, [1, batch_size]) for x in vector]

        mat = tf.dynamic_stitch([[0], [1], [2],
                                          [3], [4], [5],
                                          [6], [7], [8]], vector)
        mat = tf.reshape(mat, [3, 3, batch_size])
        mat = tf.transpose(mat, [2, 0, 1])

        return mat

    def flip_right_hand(self, coords_xyz_canonical, cond_right):
        """ Flips the given canonical coordinates, when cond_right is true. Returns coords unchanged otherwise.
                    The returned coordinates represent those of a left hand."""
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
        """Setup HandSegNet network with pre-trained model"""
        layer_per_block = [2, 2, 4, 4]
        filters_list = [64, 128, 256, 512]
        pool_list = [True, True, True, False]

        input_layer = tf.keras.layers.Input((240, 320, 3))
        x = input_layer
        for block_id, (layer, filter, pool) in enumerate(zip(layer_per_block, filters_list, pool_list), 1):
            for layer_id in range(layer):
                x = Conv_relu('conv{}_{}'.format(block_id, layer_id + 1), kernel_size=3, stride=1, filters=filter)(x)
            if pool:
                x = Max_pool('pool{}'.format(block_id))(x)
        x = Conv_relu('conv5_1', kernel_size=3, stride=1, filters=512)(x)
        x = Conv_relu('conv5_2', kernel_size=3, stride=1, filters=128)(x)

        x = Conv_relu('conv6_1', kernel_size=1, stride=1, filters=512)(x)
        x = Conv('conv6_2', kernel_size=1, stride=1, filters=2)(x)

        self.HandSegNet = tf.keras.Model(inputs=input_layer, outputs=x)
        with open('./res/weights/handsegnet-rhd.pickle', 'rb') as file:
            weight_dict = pickle.load(file)
            valid_layer_name = [key.split('/')[1] for key in weight_dict.keys()]
        # Setup params to pretrained model
        for layer in self.HandSegNet.layers:
            if layer.name not in valid_layer_name:
                continue
            weight = layer.weights
            name = 'HandSegNet/' + layer.name + '/'
            weight[0].assign(weight_dict[name + 'weights'])
            weight[1].assign(weight_dict[name + 'biases'])

    def setup_PoseNet2D(self):
        """Setup PoseNet2D network with pre-trained model"""
        layer_per_block = [2, 2, 4, 2]
        filters_list = [64, 128, 256, 512]
        pool_list = [True, True, True, False]

        input_layer = tf.keras.layers.Input((256, 256, 3))
        x = input_layer

        for block_id, (layer_num, filter_num, pool) in enumerate(zip(layer_per_block, filters_list, pool_list), 1):
            for layer_id in range(layer_num):
                x = Conv_relu('conv{}_{}'.format(block_id, layer_id + 1), kernel_size=3, stride=1, filters=filter_num)(x)
            if pool:
                x = Max_pool('pool{}'.format(block_id))(x)
        for i in range(3, 3 + 4):
            x = Conv_relu('conv4_{}'.format(i), kernel_size=3, stride=1, filters=256)(x)
        for_concat = Conv_relu('conv4_7', kernel_size=3, stride=1, filters=128)(x)
        x = Conv_relu('conv5_1', kernel_size=1, stride=1, filters=512)(for_concat)
        x = Conv_relu('conv5_2', kernel_size=1, stride=1, filters=self.J)(x)

        layers_per_unit = 5
        num_unit = 2
        for pass_id in range(6, 6 + num_unit):
            x = tf.concat([x, for_concat], 3, name='concat{}'.format(pass_id - 6))
            for rec_id in range(layers_per_unit):
                x = Conv_relu('conv{}_{}'.format(pass_id, rec_id + 1), kernel_size=7, stride=1, filters=128)(x)
            x = Conv_relu('conv{}_6'.format(pass_id), kernel_size=1, stride=1, filters=128)(x)
            x = Conv('conv{}_7'.format(pass_id), kernel_size=1, stride=1, filters=self.J)(x)

        self.PoseNet2D = tf.keras.Model(inputs=input_layer, outputs=x)
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
        """Setup PosePrior network with pre-trained model"""
        input_layer = tf.keras.layers.Input((32, 32, 21))
        x = input_layer

        filters_list = [32, 64, 128]
        for i, filter in enumerate(filters_list):
            x = Conv_relu('conv_pose_{}_1'.format(i), kernel_size=3, stride=1, filters=filter)(x)
            x = Conv_relu('conv_pose_{}_2'.format(i), kernel_size=3, stride=2, filters=filter)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.concat([x, self.hand_side], 1)

        for i, units in enumerate([512, 512]):
            x = tf.keras.layers.Dense(512, activation='relu', name='fc_rel%d' % i)(x)
            x = tf.keras.layers.Dropout(.2)(x)
        x = tf.keras.layers.Dense(self.J * 3, name='fc_xyz')(x)
        self.PosePrior = tf.keras.Model(inputs=input_layer, outputs=x)
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

    def setup_GestureNet(self):
        """setup GestureNoet network with pre-trained model"""
        n_block = 3
        strides_list = [1, 2]
        filters_list = [64, 128, 256]

        input_layer = tf.keras.layers.Input((32, 32, 21))
        x = input_layer

        for i in range(n_block):
            for j, stride in enumerate(strides_list):
                x = Conv_relu(name='conv_vp_{}_{}'.format(i, j + 1), kernel_size=3,
                              stride=stride, filters=filters_list[i])(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.concat([x, self.hand_side], 1)
        x = tf.keras.layers.Dense(256, activation='relu', name='fc_vp0')(x)
        x = tf.keras.layers.Dropout(.25)(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='fc_vp1')(x)
        x = tf.keras.layers.Dropout(.25)(x)
        ux = tf.keras.layers.Dense(1, activation='relu', name='fc_vp_ux')(x)
        uy = tf.keras.layers.Dense(1, activation='relu', name='fc_vp_uy')(x)
        uz = tf.keras.layers.Dense(1, activation='relu', name='fc_vp_uz')(x)
        self.GestureNet = tf.keras.Model(inputs=input_layer, outputs=[ux, uy, uz])

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
