import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import datetime

from data_generation import gen_2d_locations

import pdb


# !! This code is adapted from the challenge-winning repo of KarimAmer: !!
# https://github.com/radiantearth/crop-type-detection-ICLR-2020/blob/master/solutions/KarimAmer/main.py

def get_cv4a_data(data_path, experiment_name='cv4a_v1'):
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
    fields_arr = fields_arr[gts > -1]
    gts = gts[gts > -1]

    # dates
    dates = [datetime.datetime(2019, 6, 6, 8, 10, 7),
             datetime.datetime(2019, 7, 1, 8, 10, 4),
             datetime.datetime(2019, 7, 6, 8, 10, 8),
             datetime.datetime(2019, 7, 11, 8, 10, 4),
             datetime.datetime(2019, 7, 21, 8, 10, 4),
             datetime.datetime(2019, 8, 5, 8, 10, 7),
             datetime.datetime(2019, 8, 15, 8, 10, 6),
             datetime.datetime(2019, 8, 25, 8, 10, 4),
             datetime.datetime(2019, 9, 9, 8, 9, 58),
             datetime.datetime(2019, 9, 19, 8, 9, 59),
             datetime.datetime(2019, 9, 24, 8, 9, 59),
             datetime.datetime(2019, 10, 4, 8, 10),
             datetime.datetime(2019, 11, 3, 8, 10)]


    # reshape data and split into train and test
    imgs = imgs.swapaxes(2, -1)
    num_data, T_t, T_x, T_y, M = imgs.shape
    t_spatial = gen_2d_locations(T_x*T_y)
    t_spatial = (t_spatial - t_spatial.mean()) / t_spatial.std()
    t_time = np.array([(d-dates[0]).days for d in dates])
    t_time = (t_time - t_time.mean())/t_time.std()
    t_time = t_time.repeat(T_x*T_y)
    t = np.hstack((t_time[:, None], np.tile(t_spatial, (T_t, 1))))

    # two different types of experiments (see paper)
    if experiment_name == 'cv4a_v2':
        T_len = 6
        t = t[:T_len * T_x * T_y]
        t = (t - t.mean(0)[None, :]) / t.std(0)[None, :]
        imgs_tr = imgs[:, :T_len].reshape(num_data, -1, M)
        imgs_te = imgs[:, T_len:2 * T_len].reshape(num_data, -1, M)
        time_labels = np.tile(np.arange(T_len), num_data)
        return imgs_tr, imgs_te, t, time_labels
