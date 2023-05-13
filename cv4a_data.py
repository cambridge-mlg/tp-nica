import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit

import pdb


# This code is adapted from the challenge-winning repo of KarimAmer:
# https://github.com/radiantearth/crop-type-detection-ICLR-2020/blob/master/solutions/KarimAmer/main.py

def get_cv4a_data(data_path):
    imgs = np.load(os.path.join(data_path, 'imgs_13_ch_rad_16_medianxy.npy'))
    areas = np.load(os.path.join(data_path, 'areas.npy'))
    gts = np.load(os.path.join(data_path, 'gts.npy'))
    fields_arr = np.load(os.path.join(data_path, 'fields_arr.npy'))
    field_masks = np.load(os.path.join(data_path, 'field_masks_medianxy.npy'))

    # below not really needed
    tile = None

    #apply sqrt to lower skewness
    imgs = np.sqrt(imgs)

    # manually computed vegetation indices
    #generate vegitation indecies for training and testing data
    ndvi = (imgs[:,:,7:8,:,:] - imgs[:,:,3:4,:,:]) / (imgs[:,:,7:8,:,:] + imgs[:,:,3:4,:,:] + 1e-6)
    ndwi_green = (imgs[:,:,2:3,:,:] - imgs[:,:,7:8,:,:]) / (imgs[:,:,2:3,:,:] + imgs[:,:,7:8,:,:] + 1e-6)
    ndwi_blue = (imgs[:,:,1:2,:,:] - imgs[:,:,7:8,:,:]) / (imgs[:,:,1:2,:,:] + imgs[:,:,7:8,:,:] + 1e-6)

    # concatenate vegitation indicec
    imgs = np.concatenate([imgs, ndvi, ndwi_green, ndwi_blue], axis = 2)

    # standardize
    for c in range(imgs.shape[2]):
        mean = imgs[:, :, c].mean()
        std = imgs[:, :, c].std()
        imgs[:, :, c] = (imgs[:, :, c] - mean) / std


    # get tr, val, tst indices
    #sss = StratifiedShuffleSplit(n_splits=10, test_size=0.15, random_state=0)
    #
    #tr_idx, val_idx = sss.split(areas[gts > -1], gts[gts > -1])

    #idx = np.array(index)
    imgs = imgs[gts > -1]
    areas = areas[gts > -1]
    field_masks = field_masks[gts > -1]
    gts = gts[gts > -1]
    return imgs, areas, field_masks, gts




pdb.set_trace()
