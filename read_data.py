import sys
import os
import numpy as np
from scipy.misc import imread

dataFolder = './data/stage1_train/'
folders = os.listdir(dataFolder)
folders.sort()

patch_size = 256
stride = 128

images = []
masks = []
i = 0
for folder in folders:
    if folder.startswith('.'):
        continue
    imgdir = dataFolder + folder +'/images/'
    imgname = os.listdir(imgdir)
    img = imread(imgdir + imgname[0], mode = 'RGB')

    height = img.shape[0]
    width = img.shape[1]
    
    maskdir = dataFolder + folder +'/masks/'
    maskname = os.listdir(maskdir)
    mask = np.zeros((height, width))
    for m in maskname:
        mask += np.array((imread(maskdir + m, mode = 'L')))

    for h in range(0, height, stride):
        for w in range(0, width, stride):
           patch_height = min(patch_size, height - h)
           patch_width = min(patch_size, width - w)
           # img
           patch = np.zeros((256, 256, 3))
           patch[:patch_height, :patch_width, :] = img[h: h+patch_height, w: w+patch_width, :]
           images.append(patch)
           # mask
           patch = np.zeros((256, 256))
           patch[:patch_height, :patch_width] = mask[h: h+patch_height, w: w+patch_width]
           masks.append(patch)

    i += 1
    if i == 640:
        print("640 read:", images.shape)
    
images = np.array(images)
masks = np.array(masks)
print(images.shape)
print(masks.shape)
print(np.max(masks))
print(np.min(masks))
np.save('./data/images_slice.npy', np.uint8(images))
np.save('./data/masks_slice.npy', np.uint8(masks))