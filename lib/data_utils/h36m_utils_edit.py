import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3' # 指定gpu
import cv2
import glob
import h5py
import json
import joblib
import argparse
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
import scipy.io as sio

import sys
sys.path.append('.')

from lib.models import spin
from lib.core.config import TCMR_DB_DIR, BASE_DATA_DIR
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14,SMPL_MEAN_PARAMS
from lib.backbone.cliff_hr48 import CLIFF
from lib.utils.utils import tqdm_enumerate,strip_prefix_if_present
from lib.data_utils._kp_utils import convert_kps
from lib.data_utils._img_utils import get_bbox_from_kp2d
from lib.data_utils._feature_extractor import extract_features

from lib.data_utils._occ_utils import load_occluders
from lib.models.smpl import H36M_TO_J14, SMPL_MODEL_DIR, SMPL
from lib.utils.smooth_bbox import get_smooth_bbox_params, get_all_bbox_params
from lib.utils.vis import draw_skeleton
'''
直接读取human3.6m的数据，把feature部分替换掉
'''

VIS_THRESH = 0.3




def read_data_train(dataset_path, data,set='train',  debug=False):
    occluders = load_occluders("/media/DATA2/wchuq/3DHPE/dataset/voc/VOCdevkit/VOC2012/")
    '''resnet-50'''
    # load model
    device = 'cuda'
    model = CLIFF(SMPL_MEAN_PARAMS).to(device)
    print("Load the CLIFF checkpoint from path:", args.ckpt)
    state_dict = torch.load(args.ckpt)['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    # model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.load_state_dict(state_dict, strict=False)

    '''resnet'''
    # device = 'cuda'
    # model = spin.get_pretrained_hmr().to(device)
    # print("Load the CLIFF checkpoint from path:", args.ckpt)
    # state_dict = torch.load(args.ckpt)['model']
    # state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    # # model.load_state_dict(state_dict, strict=True)
    # model.eval()
    # model.load_state_dict(state_dict, strict=False)

    

    imgs_name = data['img_name']
    new_img_paths = []
    for index,img_root in enumerate(tqdm(imgs_name)):
        img_name = img_root.split('/')[-1] # 获取图片名称
        # act = str(int(img_name.split('_act_')[-1][0:2]))
        # subact = str(int(img_name.split('_subact_')[-1][0:2]))
        # cam = str(int(img_name.split('_ca_')[-1][0:2]))
        img_folder_list = img_name.split('_')[:-1]
        img_folder = '_'.join(img_folder_list)
        new_img_root  = osp.join("/media/DATA2/wchuq/3DHPE/H36M-Toolbox/images/",img_folder,img_name)
        # j_2d = data['joints2D'][index]
        # bbox = data['bbox'][index]
        new_img_paths.append(new_img_root)

    img_paths_array = np.array(new_img_paths) # 得到新的图片位置
    # 得到每组的首尾坐标
    length = len(img_paths_array)
    begin_index = np.arange(0,length,1000)
    end_index = begin_index+1000
    end_index[-1] = length-1
    features = []
    # for begin,end in tqdm(zip(begin_index,end_index),total=len(begin_index)):

    #     j_2d = data['joints2D'][begin:end]
    #     bbox = data['bbox'][begin:end]

    #     feature = extract_features(model, None, img_paths_array[begin:end], bbox,
    #                                         kp_2d = j_2d, debug=debug, dataset='h36m', scale=0.9)  # 1.2 for h36m_train_25fps_occ_db.pt
    #     features.append(feature)  

    # 五倍下采样
    vid_names = []
    frame_ids = []
    joints_3ds = []
    joints_2ds = []
    shapes = []
    poses = []
    img_names = []
    bboxs = []

    for begin,end in tqdm(zip(begin_index,end_index),total=len(begin_index)):

        vid_name = data['vid_name'][begin:end]
        frame_id = data['frame_id'][begin:end]
        joints_3d = data['joints3D'][begin:end]
        joints_2d = data['joints2D'][begin:end]
        shape = data['shape'][begin:end]
        pose = data['pose'][begin:end]
        img_name = data['img_name'][begin:end]
        bbox = data['bbox'][begin:end]

        feature = extract_features(model, occluders, img_paths_array[begin:end], bbox,
                                            kp_2d = joints_2d, debug=debug, dataset='h36m', scale=0.9)  # 1.2 for h36m_train_25fps_occ_db.pt
        
        features.append(feature)  
        vid_names.append(vid_name)  
        frame_ids.append(frame_id)  
        joints_3ds.append(joints_3d)  
        joints_2ds.append(joints_2d)  
        shapes.append(shape) 
        poses.append(pose) 
        img_names.append(img_name)
        bboxs.append(bbox)





    data['vid_name'] = np.concatenate(vid_names)
    data['frame_id'] = np.concatenate(frame_ids)
    data['joints3D'] = np.concatenate(joints_3ds)
    data['joints2D'] = np.concatenate(joints_2ds)
    data['shape'] = np.concatenate(shapes)
    data['pose'] = np.concatenate(poses)
    data['img_name'] = np.concatenate(img_names)
    data['bbox'] = np.concatenate(bboxs)
    data['feature'] = np.concatenate(features)

    dataset = data


    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default="/media/DATA2/wchuq/3DHPE/dataset/human3.6m/")
    parser.add_argument('--set', type=str, help='select train/test set', default='train')
    parser.add_argument('--ckpt', type=str, help='cliff checkpoint', default='data/base_data/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt')
    args = parser.parse_args()

    data = joblib.load ("/media/DATA2/wchuq/3DHPE/dataset/vibe_data/train/h36m_train_25fps_db.pt")
    # data = joblib.load ("/media/DATA2/wchuq/3DHPE/dataset/vibe_data/val/h36m_test_25fps_db.pt")
    # data = joblib.load("/media/DATA2/wchuq/3DHPE/dataset/vibe_data/val/h36m_test_front_25fps_tight_db.pt")
    # data = joblib.load ("/media/DATA2/wchuq/3DHPE/dataset/vibe_data/train/h36m_"+str(args.set)+"_25fps_db.pt")

    # data2['pose'] = data1['pose']

    dataset = read_data_train(args.dir, data, args.set )
    joblib.dump(dataset, osp.join("self_preprocess_data/h36m", f'hr48_h36m_{args.set}_occ_scale0.9_downsampling_db.pt'))  # h36m_train_25fps_occ_db.pt



