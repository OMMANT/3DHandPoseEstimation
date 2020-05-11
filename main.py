from __future__ import print_function, unicode_literals

import os

from network.HandPoseNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d
from utils.utils import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    img, mask = load_data(1000)
    pick_idx = np.random.choice(np.arange(img.shape[0]), 5)
    image_list = ['./res/img/' + file_name for file_name in os.listdir('./res/img') if file_name.endswith('.png')]
    image_list.extend([img[idx] for idx in pick_idx])

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
        fig = plt.figure(1, figsize=(16, 9))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224, projection='3d')
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.imshow(image)
        plot_hand(coord_hw, ax1)
        ax2.imshow(image_crop)
        plot_hand(coord_hw_crop, ax2)
        ax3.imshow(np.argmax(hand_score_map, 2))
        plot_hand_3d(keypoint_coord3d, ax4)
        ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
        ax4.set_xlim([-3, 3])
        ax4.set_ylim([-3, 1])
        ax4.set_zlim([-3, 3])
        plt.show()
