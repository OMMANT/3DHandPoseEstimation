import os, sys, cv2
import numpy as np
import tensorflow as tf

def plot_hand(coords_hw, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)

def detect_keypoints(scoremaps):
    """ Performs detection per scoremap for the hands keypoints. """
    if len(scoremaps.shape) == 4:
        scoremaps = np.squeeze(scoremaps)
    s = scoremaps.shape
    assert len(s) == 3, "This function was only designed for 3D Scoremaps."
    assert (s[2] < s[1]) and (s[2] < s[0]), "Probably the input is not correct, because [H, W, C] is expected."

    keypoint_coords = np.zeros((s[2], 2))
    for i in range(s[2]):
        v, u = np.unravel_index(np.argmax(scoremaps[:, :, i]), (s[0], s[1]))
        keypoint_coords[i, 0] = v
        keypoint_coords[i, 1] = u
    return keypoint_coords

def trafo_coords(keypoints_crop_coords, centers, scale, crop_size):
    """ Transforms coords into global image coordinates. """
    keypoints_coords = np.copy(keypoints_crop_coords)
    keypoints_coords -= crop_size // 2
    keypoints_coords /= scale
    keypoints_coords += centers

    return keypoints_coords

def single_obj_scoremap(scoremap):
    """ Applies my algorithm to figure out the most likely object from a given segmentation scoremap. """
    filter_size = 21
    s = scoremap.shape
    assert len(s) == 4, "Scoremap must be 4D"

    scoremap_softmax = tf.nn.softmax(scoremap)
    scoremap_fg = tf.reduce_max(scoremap_softmax[:, :, :, 1:], 3)
    detmap_fg = tf.round(scoremap_fg)

    max_loc = find_max_location(scoremap_fg)

    object_map_list = []
    kernel_dil = tf.ones((filter_size, filter_size, 1)) / float(filter_size * filter_size)
    for i in range(s[0]):
        # create initial objectmap (put a one at the maximum)
        sparse_ind = tf.cast(tf.reshape(max_loc[i, :], [1, 2]), tf.int64) # reshape that its one point with 2dim)
        sparse_value = [1.]
        output_shape = [s[1], s[2]]
        sTensor = tf.sparse.SparseTensor(sparse_ind, sparse_value, output_shape)
        object_map = tf.sparse.to_dense(sTensor)

        # grow the map by dilation and pixelwise and
        num_passes = max(s[1], s[2]) // (filter_size//2) # number of passes needes to make sure the map can spread over the whole image
        for j in range(num_passes):
            object_map = tf.reshape(object_map, [1, s[1], s[2], 1])
            object_map_dil = tf.compat.v1.nn.dilation2d(object_map, kernel_dil, [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')
            object_map_dil = tf.reshape(object_map_dil, [s[1], s[2]])
            object_map = tf.round(tf.multiply(detmap_fg[i, :, :], object_map_dil))

        object_map = tf.reshape(object_map, [s[1], s[2], 1])
        object_map_list.append(object_map)

    object_map = tf.stack(object_map_list)
    return object_map


def find_max_location(scoremap):
    s = scoremap.get_shape().as_list()
    if len(s) == 4:
        scoremap = tf.squeeze(scoremap, [3])
    if len(s) == 2:
        scoremap = tf.expand_dims(scoremap, 0)
    x_range = tf.expand_dims(tf.range(s[1]), 1)
    y_range = tf.expand_dims(tf.range(s[2]), 0)
    X = tf.tile(x_range, [1, s[2]])
    Y = tf.tile(y_range, [s[1], 1])

    x_vec = tf.reshape(X, [-1])
    y_vec = tf.reshape(Y, [-1])
    scoremap_vec = tf.reshape(scoremap, [s[0], -1])
    max_ind_vec = tf.math.argmax(scoremap_vec, axis=1, output_type=tf.int32)

    xy_loc = list()
    for i in range(s[0]):
        x_loc = tf.reshape(x_vec[max_ind_vec[i]], [1])
        y_loc = tf.reshape(y_vec[max_ind_vec[i]], [1])
        xy_loc.append(tf.concat([x_loc, y_loc], 0))

    xy_loc = tf.stack(xy_loc, 0)
    return xy_loc

def calc_center_bb(mask):
    mask = tf.cast(mask, tf.int32)
    mask = tf.equal(mask, 1)
    length = mask.get_shape().as_list()
    if len(length) == 4:
        mask = tf.squeeze(mask, [3])
    length = mask.get_shape().as_list()
    x_range = tf.expand_dims(tf.range(length[1]), 1)
    y_range = tf.expand_dims(tf.range(length[2]), 0)
    X = tf.tile(x_range, [1, length[2]])
    Y = tf.tile(y_range, [length[1], 1])

    bb_list = []
    center_list = []
    crop_size_list = []
    for i in range(length[0]):
        X_masked = tf.cast(tf.boolean_mask(X, mask[i, :, :]), tf.float32)
        Y_masked = tf.cast(tf.boolean_mask(Y, mask[i, :, :]), tf.float32)

        x_min = tf.reduce_min(X_masked)
        x_max = tf.reduce_max(X_masked)
        y_min = tf.reduce_min(Y_masked)
        y_max = tf.reduce_max(Y_masked)

        start = tf.stack([x_min, y_min])
        end = tf.stack([x_max, y_max])
        bb = tf.stack([start, end], 1)
        bb_list.append(bb)

        center_x = 0.5 * (x_max + x_min)
        center_y = 0.5 * (y_max + y_min)
        center = tf.stack([center_x, center_y], 0)

        center = tf.cond(tf.reduce_all(tf.math.is_finite(center)), lambda: center,
                         lambda: tf.constant([160.0, 160.0]))
        center.set_shape([2])
        center_list.append(center)

        crop_size_x = x_max - x_min
        crop_size_y = y_max - y_min
        crop_size = tf.expand_dims(tf.maximum(crop_size_x, crop_size_y), 0)
        crop_size = tf.cond(tf.reduce_all(tf.math.is_finite(crop_size)), lambda: crop_size,
                            lambda: tf.constant([100.0]))
        crop_size.set_shape([1])
        crop_size_list.append(crop_size)

    center = tf.stack(center_list)
    crop_size = tf.stack(crop_size_list)

    return center, crop_size

def crop_image_from_xy(image, crop_location, crop_size, scale):
    length = list(image.shape)

    scale = tf.reshape(scale, [-1])
    crop_location = tf.cast(crop_location, tf.float32)
    crop_location = tf.reshape(crop_location, [length[0], 2])
    crop_size = tf.cast(crop_size, tf.float32)

    crop_size_scaled = crop_size / scale
    y1 = crop_location[:, 0] - crop_size_scaled // 2
    y2 = y1 + crop_size_scaled
    x1 = crop_location[:, 1] - crop_size_scaled // 2
    x2 = x1 + crop_size_scaled
    y1 /= length[1]
    y2 /= length[1]
    x1 /= length[2]
    x2 /= length[2]
    boxes = tf.stack([y1, x1, y2, x2], -1)

    crop_size = tf.cast(tf.stack([crop_size, crop_size]), tf.int32)
    box_ind = tf.range(length[0])

    image_c = tf.image.crop_and_resize(tf.cast(image, tf.float32), boxes, box_ind, crop_size)
    return image_c

def plot_hand_3d(coords_xyz, axis, color_fixed=None, linewidth='1'):
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_xyz[connection[0], :]
        coord2 = coords_xyz[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color_fixed, linewidth=linewidth)

    axis.view_init(azim=-90., elev=90.)
