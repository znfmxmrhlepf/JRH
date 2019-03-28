import random
import numpy as np
import os
import glob
import cv2

def get_images(paths):
    images = np.zeros((len(paths), 600, 600, 3))
    labels = np.zeros((len(paths), 19))

    for i, path in enumerate(paths):
        images[i, :, :, :] = cv2.imread(path)
        labels[i] = path[:path.find('_')]

    return (images, labels)

def load_data():

    os.chdir('..')
    f = open('Images/list.txt')
    classes = map(lambda x:  x[:-1], f.readlines())

    paths = []
    for classf in classes:
        paths += glob.glob('Images/' + classf + '/*')

    imagePaths = [os.path.basename(x) for x in paths]
    random.shuffle(imagePaths)
    
    test_paths = imagePaths[:500]
    train_paths = imagePaths[500:]

    (test_images, test_labels) = get_images(test_paths)
    (train_images, train_labels) = get_images(train_paths)

    print(test_labels)

if __name__ == '__main__':
    load_data()
