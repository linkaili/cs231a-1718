from os import listdir
from os.path import join, splitext
import os
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import glob
from skimage.filters import threshold_otsu
from skimage.exposure import histogram
from skimage.color import rgb2gray
from scipy import ndimage
from skimage import measure
import pandas as pd
from skimage.feature import canny, peak_local_max
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage import img_as_float, img_as_uint


test_path = 'stage1_test'
train_path = 'stage1_train'
train_ids = listdir(train_path)
test_ids = listdir(test_path)

# get test and training path
test_paths = [glob.glob(join(test_path, test_id, 'images', '*'))[0] for test_id in test_ids]
train_paths = [glob.glob(join(train_path, train_id, 'images', '*'))[0] for train_id in train_ids]

# prepare path to save processed images
test_processed_paths = [join(test_path, test_id, 'images', 'processed_img.png') for test_id in test_ids]
test_gradient_paths = [join(test_path, test_id, 'images', 'gradient_img.png') for test_id in test_ids]
train_processed_paths = [join(train_path,train_id, 'images', 'processed_img.png') for train_id in train_ids]
train_gradient_paths = [join(train_path, train_id, 'images', 'gradient_img.png') for train_id in train_ids]


# randomly select an image
numimg = 2
img_raw = imread(train_paths[numimg])
#img_raw = imread('stage1_train/4e07a653352b30bb95b60ebc6c57afbc7215716224af731c51ff8d430788cd40/images/4e07a653352b30bb95b60ebc6c57afbc7215716224af731c51ff8d430788cd40.png')
#img_raw = imread('stage1_test/4f949bd8d914bbfa06f40d6a0e2b5b75c38bf53dbcbafc48c97f105bee4f8fac/images/4f949bd8d914bbfa06f40d6a0e2b5b75c38bf53dbcbafc48c97f105bee4f8fac.png')
if len(img_raw.shape) > 2:
    img_gray = rgb2gray(img_raw)
else:
    img_gray = img_raw
img_gray = 255./np.max(img_gray)*img_gray
print 'maximum pixel intensity', np.max(img_gray)

plt.figure
plt.subplot(121)
plt.imshow(img_raw)
plt.axis('off')

print 'The gray image dimenion is:', img_gray.shape
plt.subplot(122)
plt.imshow(img_gray, cmap='gray')
plt.axis('off')


# In[33]:


# doing thresholding
thresh = threshold_otsu(img_gray)
img_otsu = np.where(img_gray>thresh, 1, 0)

peripheral0 = np.sum(img_otsu[0,:] == 0) + np.sum(img_otsu[:, 0] == 0) + np.sum(img_otsu[:, -1] == 0)+ np.sum(img_otsu[-1, :] == 0)
peripheral1 = np.sum(img_otsu[0,:] == 1) + np.sum(img_otsu[:, 0] == 1) + np.sum(img_otsu[:, -1] == 1)+ np.sum(img_otsu[-1, :] == 1)


# print np.sum(img_otsu == 0)
# print np.sum(img_otsu == 1)
# print peripheral0, peripheral1

# some cases the background would be brighter than the nuclei, which results in background to be 1
if peripheral0 < peripheral1:
    print 'flipped'
    img_otsu = np.where(img_gray>thresh, 0, 1)
    img_gray = 255 - img_gray

plt.figure()
plt.imshow(img_otsu, cmap= 'gray')
plt.title('Otsu Segmented Image')

# calculate the gradient map
elevation_map = sobel(img_gray)

edge_thresh = threshold_otsu(elevation_map)
edge_otsu = np.where(elevation_map > edge_thresh, 1, 0)
plt.figure()
plt.imshow(edge_otsu, cmap = 'gray')
plt.title('edges after binary fill holes')

# combine edges and otsu thresholded images
img = np.logical_or(img_otsu, edge_otsu)
img = ndi.binary_fill_holes(img)
# # open up some overlaped cells
img = ndimage.binary_erosion(img)

# separate different cells
distance = ndi.distance_transform_edt(img)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((15, 15)),
                            labels=img)
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=img)


plt.figure()
plt.imshow(labels)
plt.title('Label Image')

# plt.figure()
# plt.imshow(img_segmented, cmap = 'gray')
# plt.title('Segmented Image')

# # img = np.multiply(img_gray, img_segmented)
# plt.figure()
# plt.imshow(img, cmap = 'gray')
# plt.title('Diff')

# imsave(test_processed_paths[numimg], img_as_float(img/255.))
# imsave(test_gradient_paths[numimg], edge_otsu)
# hist, bins_center = histogram(img_gray)
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(bins_center, hist, color='b')
# plt.fill_between(bins_center, 0, hist, color = 'b')
# ax.axvline(thresh, color='k', ls='--')


# In[35]:


# labels, nlabels = measure.label(labels, return_num=True)
plt.figure()
plt.imshow(labels, cmap='spectral')
print 'The segmented number of labels is:', nlabels

# aggregate masks into a single one
ground_truth = np.zeros(img_gray.shape, dtype = float)
train_mask_paths = glob.glob(join(train_path, train_ids[numimg], 'masks', '*'))
print 'ground truth number of labels:', len(train_mask_paths)
ground_truth_label = 0
for mask_path in train_mask_paths:
    ground_truth_label += 1
    mask = imread(mask_path)
    mask = ground_truth_label*mask
    ground_truth = np.maximum(ground_truth, mask)

print labels.max()
# imsave('processed2.png', labels)

np.save('processed1', labels)

plt.figure(figsize = (6,8))
plt.subplot(131)
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.title('Original Image')

plt.subplot(132)
plt.imshow(labels, cmap='spectral')
plt.axis('off')
plt.title('Prediction')

plt.subplot(133)
plt.imshow(ground_truth, cmap='spectral')
plt.axis('off')
plt.title('Ground Truth')
print 'The segmented number of labels after binary opening:', nlabels
print ground_truth_label


# In[9]:


# generate individual mask for each cell
masks = [np.where(i+1==labels, 1, 0) for i in range(nlabels)]
# run-line encoding
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1):
            run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])

print rle_encoding(masks[0])


# In[43]:


# wrap up the previous functions
def read_in_imgs():
    '''
    This function reads in images and form file path to access those images
    '''
    # import images
    test_path = 'stage1_test'
    train_path = 'stage1_train'

    train_ids = listdir(train_path)
    test_ids = listdir(test_path)

    # get test and training path
    test_paths = [glob.glob(join(test_path, test_id, 'images', '*'))[0] for test_id in test_ids]
    train_paths = [glob.glob(join(train_path, train_id, 'images', '*'))[0] for train_id in train_ids]

def analyze_imgs(path):
    '''
    Input: list of path to one image
    Output: The segmented nuclei masks for the image
    '''
    img_raw = imread(path)
    #img_raw = image
    if len(img_raw.shape) > 2:
        img_gray = rgb2gray(img_raw)
    else:
        img_gray = img_raw
    # normalize every image to be in [0, 255]
    img_gray = 255./np.max(img_gray)*img_gray

    # do otsu thresholding
    thresh = threshold_otsu(img_gray)
    img_otsu = np.where(img_gray>thresh, 1, 0)

    peripheral0 = np.sum(img_otsu[0,:] == 0) + np.sum(img_otsu[:, 0] == 0) + np.sum(img_otsu[:, -1] == 0)    + np.sum(img_otsu[-1, :] == 0)
    peripheral1 = np.sum(img_otsu[0,:] == 1) + np.sum(img_otsu[:, 0] == 1) + np.sum(img_otsu[:, -1] == 1)    + np.sum(img_otsu[-1, :] == 1)


    # print np.sum(img_otsu == 0)
    # print np.sum(img_otsu == 1)
    # print peripheral0, peripheral1

    # some cases the background would be brighter than the nuclei, which results in background to be 1
    if peripheral0 < peripheral1:
        print 'flipped'
        img_otsu = np.where(img_gray>thresh, 0, 1)
        img_gray = 255 - img_gray

#     # watershed method
#     markers = np.zeros(img_gray.shape)

#     markers[img_gray < 30] = 1
#     markers[img_gray > 120] = 2
#     elevation_map = sobel(img_gray)
#     edge_thresh = threshold_otsu(elevation_map)
#     edge_otsu = np.where(elevation_map > edge_thresh, 1, 0)

#     # make sure the edge values are higher values
#     if np.sum(edge_otsu == 0) < np.sum(edge_otsu==1):
#         edge_otsu = np.where(elevation_map > edge_thresh, 0, 1)

#     # img_segmented2 = np.logical_or(img_segmented, watershed(elevation_map, markers)-1)
#     img_segmented2 = watershed(elevation_map, markers)
#     # some cases the background would be brighter than the nuclei, which results in background to be 1
#     if np.sum(img_segmented2 -1 == 0) < np.sum(img_segmented2- 1 == 1):
#         print 'flipped'
#         img_otsu = 1 - img_otsu
#         img_segmented2 = 2 - img_segmented2
        # img_segmented2 = ndi.binary_fill_holes(img_segmented2-1)
    elevation_map = sobel(img_gray)

    edge_thresh = threshold_otsu(elevation_map)
    edge_otsu = np.where(elevation_map > edge_thresh, 1, 0)
#     plt.figure()
#     plt.imshow(edge_otsu, cmap = 'gray')
#     plt.title('edges after binary fill holes')

    # combine edges and otsu thresholded images
    img = np.logical_or(img_otsu, edge_otsu)
    img = ndi.binary_fill_holes(img)
    # # open up some overlaped cells
    img = ndimage.binary_erosion(img)

    # separate different cells
    distance = ndi.distance_transform_edt(img)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((20, 20)),
                                labels=img)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=img)
#     img_segmented = np.logical_or(img_otsu, img_segmented2)
#     # combine with sobel edges
#     img_segmented = np.logical_or(img_segmented, edge_otsu)
#     img = np.multiply(img_gray, img_segmented)
#     print 'max intensity:', img.max()

#     imsave(process_path, img/img.max())

#     # open up some overlaped cells
#     img_opened= ndimage.binary_opening(img_processed, iterations=1)
    return labels


# In[44]:


# prepare submission ready file
if __name__ == "__main__":
    read_in_imgs
    result = pd.DataFrame()
    for n, test_id in enumerate(test_ids):
        print n
    #     print test_paths[n]
    #     print test_processed_paths[n]
        masks = analyze_imgs(test_paths[n])
        for mask in masks:
            rle = rle_encoding(mask)
            s = pd.Series({'ImageId': test_id, 'EncodedPixels': rle})
            result = result.append(s, ignore_index=True)


# In[42]:


print result.sample(5)
print 'The total number fo masks is:', result.size
result.to_csv('submission.csv', index=None)


# In[ ]:


iteration_of_opening = [1,2,3,4,5,6]
scores = [0.244, 0.234, 0.216, 0.177, 0.134,0.112]
plt.plot(iteration_of_opening, scores, linewidth = 2)
plt.xlabel('# of Iterations of Binary Opening', fontsize = 16)
plt.ylabel('Scores', fontsize = 16)
plt.title('Influence of Binary Opening Iterations on Final Score', fontsize = 16)
plt.show()


# In[ ]:

# for
results = np.zeros(images.shape[:3])
for i in range(images.shape[0]):
    print i
    results[i,:,:] = analyze_imgs(images[i,:,:])
plt.figure
plt.imshow(results[2,:,:], cmap = 'gray')
plt.show()
np.save('processed_val',results)
