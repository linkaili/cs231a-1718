import sys
import os
import numpy as np
from scipy.misc import imread

s = (256, 256, 3)

patch_size = 256

dataFolder = './data/stage1_test/'
folders = os.listdir(dataFolder)
folders.sort()
# fileNames.sort()

images = []

ids = []
for folder in folders:
    if folder.startswith('.'):
        continue
    # print len(images)

    imgdir = dataFolder + folder +'/images/'
    imgname = os.listdir(imgdir)
    img = imread(imgdir + imgname[0], mode = 'RGB')
    # images.append(img[:256, :256, :])
    height = img.shape[0]
    num_h = np.ceil(height*1.0/patch_size).astype(np.int32)
    # h_pix = np.ceil(height / num_h).astype(np.int32)
    width = img.shape[1]
    num_w = np.ceil(width*1.0/patch_size).astype(np.int32)
    # w_pix = np.ceil(width / num_w).astype(np.int32)

    for h in range(0, height, patch_size/2):
        for w in range(0, width, patch_size/2):
           patch = np.zeros((256, 256, 3))
           patch_height = min(patch_size, height - h)
           patch_width = min(patch_size, width - w) 
           patch[:patch_height, :patch_width, :] = \
           img[h: h+patch_height, w: w+patch_width, :]
           images.append(patch)

    
images = np.array(images)
print images.shape

np.save('test_images2.npy', np.uint8(images))
