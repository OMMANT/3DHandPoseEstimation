from __future__ import print_function, unicode_literals

import os

from network.HandPoseNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords
from utils.utils import *


if __name__ == '__main__':
    image_list = ['./res/img/' + file_name for file_name in os.listdir('./res/img') if file_name.endswith('.png')]

    net = ColorHandPose3DNetwork()

    for image_name in image_list:
        image = cv2.imread(image_name) if isinstance(image_name, str) else image_name
        image = cv2.resize(image, (320, 240))
        image_v = np.expand_dims((image.astype(np.float32) / 255.) - .5, axis=0)

        inference = net.inference(image_v)
        hand_score_map, image_crop, scale, center, keypoint_score_map, keypoint_coord3d = tuple(inference)

        hand_score_map = np.squeeze(hand_score_map, axis=0) # (1, 256, 256, 2) -> (256, 256, 2)
        image_crop = np.squeeze(image_crop) # (1, 256, 256, 3) -> (256, 256, 3)
        keypoint_score_map = np.squeeze(keypoint_score_map) # (1, 256, 256, 21) -> (256, 256, 21)
        keypoint_coord3d = np.squeeze(keypoint_coord3d) # (1, 21, 3) -> (21, 3)

        image_crop = ((image_crop + .5) * 255).astype(np.uint8)
        coord_hw_crop = detect_keypoints(np.squeeze(keypoint_score_map))
        coord_hw = trafo_coords(coord_hw_crop, center, scale, 256)

        # visualize
        plot_inference(image, image_crop, coord_hw, coord_hw_crop, hand_score_map, keypoint_coord3d)
