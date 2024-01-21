'''
mpii_3d提取帧
'''
import argparse
import os
import pickle
import shutil
from os.path import join

import cv2
import h5py
import mmcv
import numpy as np
from scipy.io import loadmat

train_subjects = [i for i in range(8, 9)]
test_subjects = [i for i in range(1, 7)]
train_seqs = [1, 2]
train_cams = [0, 1, 2, 4, 5, 6, 7, 8]
train_frame_nums = {
    (1, 1): 6416,
    (1, 2): 12430,
    (2, 1): 6502,
    (2, 2): 6081,
    (3, 1): 12488,
    (3, 2): 12283,
    (4, 1): 6171,
    (4, 2): 6675,
    (5, 1): 12820,
    (5, 2): 12312,
    (6, 1): 6188,
    (6, 2): 6145,
    (7, 1): 6239,
    (7, 2): 6320,
    (8, 1): 6468,
    (8, 2): 6054
}
test_frame_nums = {1: 6151, 2: 6080, 3: 5838, 4: 6007, 5: 320, 6: 492}
train_img_size = (2048, 2048)
root_index = 14
joints_17 = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]


def load_trainset(data_root, out_dir):
    """Load training data, create annotation file and camera file.
    Args:
        data_root: Directory of dataset, which is organized in the following
            hierarchy:
                data_root
                |-- train
                    |-- S1
                        |-- Seq1
                        |-- Seq2
                    |-- S2
                    |-- ...
                |-- test
                    |-- TS1
                    |-- TS2
                    |-- ...
        out_dir: Directory to save annotation file.
    """
    _imgnames = []

    img_dir = join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    for subj in train_subjects:
        for seq in train_seqs:
            
            seq_path = join(data_root,  f'S{subj}', f'Seq{seq}')
            num_frames = train_frame_nums[(subj, seq)]

            for cam in train_cams:
                os.makedirs(join(img_dir,f'S{subj}',f'Seq{seq}',f'video_{cam}'), exist_ok=True) # f:把{}里的转为数字
                # extract frames from video
                # 提取帧信息
                video_path = join(seq_path, 'imageSequence',
                                  f'video_{cam}.avi')
                video = mmcv.VideoReader(video_path)
                for i in mmcv.track_iter_progress(range(num_frames)):
                    img = video.read()
                    if img is None:
                        break
                    imgname = f'S{subj}_Seq{seq}_Cam{cam}_{i+1:06d}.jpg'
                    _imgnames.append(imgname)
                    cv2.imwrite(join(img_dir,f'S{subj}',f'Seq{seq}',f'video_{cam}', imgname), img)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('data_root', type=str, help='data root', default="/media/DATA2/wchuq/3DHPE/dataset/3dhp/")
    # parser.add_argument(
    #     'out_dir', type=str, help='directory to save annotation files.',default="/media/DATA2/wchuq/3DHPE/dataset/3dhp/")
    # args = parser.parse_args()
    # data_root = args.data_root
    # out_dir = args.out_dir
    data_root = "/media/DATA2/wchuq/3DHPE/dataset/3dhp/"
    out_dir = "/media/DATA2/wchuq/3DHPE/dataset/3dhp/"

    load_trainset(data_root, out_dir)
