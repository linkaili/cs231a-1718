from skimage.morphology import label # label regions
import os
import pandas as pd
import numpy as np
from scipy import ndimage
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cut_off = 0.5):
    lab_img = label(x>cut_off)
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)



def analyze_image(pred_test_i, new_test_ids_i):
    im_id = new_test_ids_i
    mask = pred_test_i

    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)    
    labels, nlabels = ndimage.label(mask)
    print nlabels
    
    im_df = pd.DataFrame()
    # if nlabels == 0:
    #     s = pd.Series({'ImageId': im_id, 'EncodedPixels': []})
    #     im_df = im_df.append(s, ignore_index=True) 

    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            s = pd.Series({'ImageId': im_id, 'EncodedPixels': rle})
            im_df = im_df.append(s, ignore_index=True)   
    return im_df

def analyze_list_of_images(pred_test, new_test_ids):
    
    all_df = pd.DataFrame()
    for i in range(len(pred_test)):
        im_df = analyze_image(pred_test[i], new_test_ids[i])
        all_df = all_df.append(im_df, ignore_index=True)
    return all_df

if __name__ == '__main__':

    dataFolder = './data/stage1_test/'
    folders = os.listdir(dataFolder)
    folders.sort()
    new_test_ids = folders[1:]
    pred_test = np.load('pred_test.npy')[:-1,...].astype(np.int32)/255
    print len(new_test_ids)
    print pred_test.shape
    df = analyze_list_of_images(pred_test, new_test_ids)
    df.to_csv('submission.csv', index=None)

