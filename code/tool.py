import random
import time
import numpy as np
import os
import glob
import cv2

random.seed(time.time())

def get_images(paths):
    images = np.zeros((len(paths), 60, 60, 3))
    labels = np.zeros((len(paths), 19))

    for i, path in enumerate(paths):
        images[i, :, :, :] = cv2.resize(cv2.imread(path), None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        print(path[:path.find('_')])
        labels[i] = int(path[:path.find('_')])

    return (images, labels)

def load_data():

    os.chdir('..')
    f = open('Images/list.txt')
    classes = list(map(lambda x:  x[:-1], f.readlines()))
    classes.sort()

    paths = []
    for classf in classes:
        paths += glob.glob('Images/' + classf + '/*')

    imagePaths = [os.path.basename(x) for x in paths]
    random.shuffle(imagePaths)
    
    data_test = data(imagePaths[:500])
    data_train = data(imagePaths[500:])

    return data_test, data_train

class data():
    def __init__(self, paths):
        self.paths = paths

    def get_batch(self, size):
        random.shuffle(self.paths)
        return get_images(self.paths[:size])

if __name__ == '__main__':
    load_data()
