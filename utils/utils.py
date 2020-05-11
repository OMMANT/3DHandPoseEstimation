import os, sys, cv2
import numpy as np
import matplotlib.pyplot as plt
from .general import plot_hand, plot_hand_3d
from mpl_toolkits.mplot3d import Axes3D

def printProgress(iteration, total, prefix='Progress', suffix='Complete', decimals=2, bar_length=35):
    formatStr = '{0:.' + str(decimals) + 'f}'
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(bar_length * iteration / float(total)))
    bar = '#' * filledLength + '-' * (bar_length - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n\n')
    sys.stdout.flush()

def load_data(maximum=41258):
    maximum = maximum - 1
    train_X_img_list = []
    train_Y_img_list = []
    for idx, file in enumerate(os.listdir('./res/training/color')):
        if idx > maximum:
            break
        printProgress(idx, maximum)
        train_X_img_list.append(cv2.resize(cv2.imread('./res/training/color/' + file), (320, 240)))
    for idx, file in enumerate(os.listdir('./res/training/mask')):
        if idx > maximum:
            break
        printProgress(idx, maximum)
        train_Y_img_list.append(cv2.resize(cv2.imread('./res/training/mask/' + file, cv2.IMREAD_GRAYSCALE), (320, 240)))
    return np.array(train_X_img_list), np.array(train_Y_img_list)

def load_val_data():
    maximum = 2728 - 1
    val_X = []
    val_Y = []
    for idx, file in enumerate(os.listdir('./res/evaluation/color')):
        if idx > maximum:
            break
        printProgress(idx, maximum)
        val_X.append(cv2.resize(cv2.imread('./res/evaluation/color/' + file), (320, 240)))
    for idx, file in enumerate(os.listdir('./res/evaluation/mask')):
        if idx > maximum:
            break
        printProgress(idx, maximum)
        val_Y.append(cv2.resize(cv2.imread('./res/evaluation/mask/' + file), (320, 240)))
    return np.array(val_X), np.array(val_Y)

def plot_inference(image, image_crop, coord_hw, coord_hw_crop, hand_score_map, key_point_coord3d):
    fig = plt.figure(figsize=(16, 9))
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
    plot_hand_3d(key_point_coord3d, ax4)
    ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
    ax4.set_xlim([-3, 3])
    ax4.set_ylim([-3, 1])
    ax4.set_zlim([-3, 3])
    plt.show()

