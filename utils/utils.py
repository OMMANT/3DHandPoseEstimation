import os, sys, cv2
import numpy as np

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

def multiply(lst):
    i = 1
    for element in lst:
        i *= element
    return i