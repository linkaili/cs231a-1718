import numpy as np
from PIL import Image 
import sys
import os
from scipy.misc import imread
from encode import *

preds = np.load("./pred_test2.npy")
# preds = np.load("./data/test_images2.npy")
print (np.min(preds))
print (np.max(preds))
preds = ((preds)).astype('uint8')


# for i in range(preds.shape[0]):
# 	Image.fromarray(np.squeeze(preds[i])).save('test_pred/output' + str(i) + '.png')
	


patch_size = 256
stride = patch_size/2

dataFolder = './data/stage1_test/'
folders = os.listdir(dataFolder)
folders.sort()
new_test_ids = folders[1:]
images = []
i = 0

for folder in folders:
    if folder.startswith('.'):
        continue
    # print len(images)

    imgdir = dataFolder + folder +'/images/'
    imgname = os.listdir(imgdir)
    img = imread(imgdir + imgname[0], mode = 'L')

    img_combined = np.zeros(img.shape).astype('uint8')
    height = img.shape[0]
    width = img.shape[1]
    tmp1 = min(height, 256)
    tmp2 = min(width, 256)

    for h in range(0, height, stride):
        for w in range(0, width, stride):
            patch_height = min(256, height - h)
            patch_width = min(256, width - w) 
            patch = preds[i, :patch_height, :patch_width]
            if w > 0:
                oldpatch1 = img_combined[h: h+patch_height, w: w+stride]
                oldpatch1_and = np.logical_and(oldpatch1 == 255, patch[:patch_height, :stride] == 255)
                oldpatch1[oldpatch1_and] = 255
                patch[:patch_height, :stride] = oldpatch1
            if h > 0:
                oldpatch2 = img_combined[h: h+stride, w: w+patch_width]
                oldpatch2_and = np.logical_and(oldpatch2 == 255, patch[:stride, :patch_width] == 255)
                oldpatch2[oldpatch2_and] = 255
                patch[:stride, :patch_width] = oldpatch2
            img_combined[h: h+patch_height, w: w+patch_width] = patch
            Image.fromarray(np.squeeze(preds[i])).save('test_pred/output' + str(i) + '.png')
            i += 1 


    images.append(img_combined)
    Image.fromarray(np.squeeze(img_combined)).save('test_pred/output' + str(i) + '_.png')
    # break

df = analyze_list_of_images(images, new_test_ids)
df.to_csv('submission.csv', index=None)   
# images = np.array(images)
# print images.shape

# np.save('test_images2_combined.npy', np.uint8(images))
