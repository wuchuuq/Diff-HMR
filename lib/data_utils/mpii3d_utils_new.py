import os
import cv2
import glob
import h5py
import json
import torch
import joblib
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import scipy.io as sio

import sys
sys.path.append('.')

from lib.models import spin
from lib.core.config import TCMR_DB_DIR
from lib.backbone.cliff_hr48 import CLIFF
from lib.utils.utils import tqdm_enumerate
from lib.models.smpl import SMPL_MEAN_PARAMS
from lib.data_utils._kp_utils import convert_kps
from lib.data_utils._img_utils import get_bbox_from_kp2d
from lib.data_utils._feature_extractor import extract_features
from lib.utils.utils import strip_prefix_if_present

# from lib.data_utils._occ_utils import load_occluders
'''
加上smpl版本
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 指定gpu


def read_train_data(dataset_path, smpl_data, debug=False):
    h, w = 2048, 2048
    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'shape': [],
        'pose': [],
        'bbox': [],
        'img_name': [],
        'features': [],
    }

    # occluders = load_occluders('./data/VOC2012')

    # model = spin.get_pretrained_hmr()

    # load model
    device = 'cuda'
    model = CLIFF(SMPL_MEAN_PARAMS).to(device)
    print("Load the CLIFF checkpoint from path:", args.ckpt)
    state_dict = torch.load(args.ckpt)['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.load_state_dict(state_dict, strict=False)

    # training data
    # 先处理前三个
    user_list = range(1, 9) # Subject
    seq_list = range(1, 3) # seq_i
    vid_list = range(1, 9)

    # product = product(user_list, seq_list, vid_list)
    # user_i, seq_i, vid_i = product[process_id]

    for user_i in user_list:
        print("Subject: ", user_i)
        for seq_i in seq_list:
            print("seq_i: ", seq_i)
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i),
                                    'imageSequence')
            # mat file with annotations
            annot_file = os.path.join(dataset_path, 'S' + str(user_i),
                                    'Seq' + str(seq_i),'annot.mat')
            annot2 = sio.loadmat(annot_file)['annot2']
            annot3 = sio.loadmat(annot_file)['annot3']
            # calibration file and camera parameters
            for j, vid_i in enumerate(vid_list):
                print("vid_i: ", vid_i)
                # image folder
                imgs_path = os.path.join(seq_path,
                                         'img_' + str(vid_i))
                # per frame
                pattern = imgs_path + '*.jpg'
                img_list = sorted(glob.glob(pattern)) # 获取所有的匹配路径
                vid_used_frames = []
                vid_used_joints = []
                vid_used_bbox = []
                vid_segments = []
                vid_uniq_id = "subj" + str(user_i) + '_seq' + str(seq_i) + "_vid" + str(vid_i) + "_seg0"
                for i, img_i in tqdm_enumerate(img_list): # i: frame

                    # for each image we store the relevant annotations
                    img_name = img_i.split('/')[-1]
                    joints_2d_raw = np.reshape(annot2[vid_i][0][i], (1, 28, 2))
                    joints_2d_raw= np.append(joints_2d_raw, np.ones((1,28,1)), axis=2)
                    joints_2d = convert_kps(joints_2d_raw, "mpii3d",  "spin").reshape((-1,3))

                    joints_3d_raw = np.reshape(annot3[vid_i][0][i], (1, 28, 3)) / 1000
                    joints_3d = convert_kps(joints_3d_raw, "mpii3d", "spin").reshape((-1,3))

                    bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)

                    joints_3d = joints_3d - joints_3d[39]  # 4 is the root

                    # smpl annot
                    smpl_param = smpl_data[str(user_i)][str(seq_i)][str(int(img_name.split("_")[-1].split(".")[0]))] 
                    pose = np.array(smpl_param['pose'])# 72
                    shape = np.array(smpl_param['shape']) # 10
          

                    # check that all joints are visible
                    x_in = np.logical_and(joints_2d[:, 0] < w, joints_2d[:, 0] >= 0)
                    y_in = np.logical_and(joints_2d[:, 1] < h, joints_2d[:, 1] >= 0)
                    ok_pts = np.logical_and(x_in, y_in)
                    if np.sum(ok_pts) < joints_2d.shape[0]:
                        vid_uniq_id = "_".join(vid_uniq_id.split("_")[:-1])+ "_seg" +\
                                          str(int(dataset['vid_name'][-1].split("_")[-1][3:])+1)
                        continue


                    visualize = True
                    if visualize == True and i > 500:
                        import matplotlib.pyplot as plt

                        frame = cv2.cvtColor(cv2.imread(img_i), cv2.COLOR_BGR2RGB)

                        for k in range(49):
                            kp = joints_2d[k]

                            frame = cv2.circle(
                                frame.copy(),
                                (int(kp[0]), int(kp[1])),
                                thickness=3,
                                color=(255, 0, 0),
                                radius=5,
                            )

                            cv2.putText(frame, f'{k}', (int(kp[0]), int(kp[1]) + 1), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                        (0, 255, 0),
                                        thickness=3)

                        # cv2.imshow('vis', frame)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        # cv2.waitKey(1)
                        cv2.imwrite('vis.png', frame)

                    dataset['vid_name'].append(vid_uniq_id)
                    dataset['frame_id'].append(str(int(img_name.split("_")[-1].split(".")[0])))
                    dataset['img_name'].append(img_i)
                    dataset['joints2D'].append(joints_2d)
                    dataset['joints3D'].append(joints_3d)
                    dataset['bbox'].append(bbox)
                    dataset['shape'].append(shape)
                    dataset['pose'].append(pose)
                    vid_segments.append(vid_uniq_id)
                    vid_used_frames.append(img_i)
                    vid_used_joints.append(joints_2d)
                    vid_used_bbox.append(bbox)

                vid_segments= np.array(vid_segments)
                ids = np.zeros((len(set(vid_segments))+1))
                ids[-1] = len(vid_used_frames) + 1
                if (np.where(vid_segments[:-1] != vid_segments[1:])[0]).size != 0:
                    ids[1:-1] = (np.where(vid_segments[:-1] != vid_segments[1:])[0]) + 1

                for i in tqdm(range(len(set(vid_segments)))):
                    features = extract_features(model, None, np.array(vid_used_frames)[int(ids[i]):int(ids[i+1])],
                                                vid_used_bbox[int(ids[i]):int((ids[i+1]))],
                                                kp_2d=np.array(vid_used_joints)[int(ids[i]):int(ids[i+1])],
                                                dataset='spin', debug=False, scale=1.2)
                    dataset['features'].append(features)

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
    dataset['features'] = np.concatenate(dataset['features'])

    return dataset


def read_test_data(dataset_path):

    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'bbox': [],
        'img_name': [],
        'features': [], 
        "valid_i": []
    }

    # model = spin.get_pretrained_hmr()

     # load model
    device = 'cuda'
    model = CLIFF(SMPL_MEAN_PARAMS).to(device)
    print("Load the CLIFF checkpoint from path:", args.ckpt)
    state_dict = torch.load(args.ckpt)['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.load_state_dict(state_dict, strict=False)
    user_list = range(1, 7)

    for user_i in user_list:
        print('Subject', user_i)
        seq_path = os.path.join(dataset_path,
                                'mpi_inf_3dhp_test_set',
                                'mpi_inf_3dhp_test_set',
                                'TS' + str(user_i))
        # mat file with annotations
        annot_file = os.path.join(seq_path, 'annot_data.mat')
        mat_as_h5 = h5py.File(annot_file, 'r')
        annot2 = np.array(mat_as_h5['annot2'])
        annot3 = np.array(mat_as_h5['univ_annot3'])
        valid = np.array(mat_as_h5['valid_frame'])

        vid_used_frames = []
        vid_used_joints = []
        vid_used_bbox = []
        vid_segments = []
        vid_uniq_id = "subj" + str(user_i) + "_seg0"

        for frame_i, valid_i in tqdm(enumerate(valid)):
            img_i = os.path.join('mpi_inf_3dhp_test_set',
                                    'mpi_inf_3dhp_test_set',
                                    'TS' + str(user_i),
                                    'imageSequence',
                                    'img_' + str(frame_i + 1).zfill(6) + '.jpg')

            joints_2d_raw = np.expand_dims(annot2[frame_i, 0, :, :], axis = 0)
            joints_2d_raw = np.append(joints_2d_raw, np.ones((1, 17, 1)), axis=2)


            joints_2d = convert_kps(joints_2d_raw, src="mpii3d_test", dst="spin").reshape((-1, 3))

            visualize = True
            if visualize == True:
                frame = cv2.cvtColor(cv2.imread(os.path.join(dataset_path, img_i)), cv2.COLOR_BGR2RGB)

                for k in range(49):
                    kp = joints_2d[k]

                    frame = cv2.circle(
                        frame.copy(),
                        (int(kp[0]), int(kp[1])),
                        thickness=3,
                        color=(255, 0, 0),
                        radius=5,
                    )

                    cv2.putText(frame, f'{k}', (int(kp[0]), int(kp[1]) + 1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0),
                                thickness=3)

                # cv2.imshow(f'frame:{frame_i}', frame)
                # cv2.waitKey(0)
                cv2.imwrite("self_preprocess_data/mpii3d/img.png",frame)
                # cv2.destroyAllWindows()
                # cv2.waitKey(1)


            joints_3d_raw = np.reshape(annot3[frame_i, 0, :, :], (1, 17, 3)) / 1000
            joints_3d = convert_kps(joints_3d_raw, "mpii3d_test", "spin").reshape((-1, 3))
            joints_3d = joints_3d - joints_3d[39] # substract pelvis zero is the root for test

            bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)

            # check that all joints are visible
            img_file = os.path.join(dataset_path, img_i)
            I = cv2.imread(img_file)
            h, w, _ = I.shape
            x_in = np.logical_and(joints_2d[:, 0] < w, joints_2d[:, 0] >= 0)
            y_in = np.logical_and(joints_2d[:, 1] < h, joints_2d[:, 1] >= 0)
            ok_pts = np.logical_and(x_in, y_in)

            if np.sum(ok_pts) < joints_2d.shape[0]:
                vid_uniq_id = "_".join(vid_uniq_id.split("_")[:-1]) + "_seg" + \
                              str(int(dataset['vid_name'][-1].split("_")[-1][3:]) + 1)
                continue

            dataset['vid_name'].append(vid_uniq_id)
            dataset['frame_id'].append(img_file.split("/")[-1].split(".")[0])
            dataset['img_name'].append(img_file)
            dataset['joints2D'].append(joints_2d)
            dataset['joints3D'].append(joints_3d)
            dataset['bbox'].append(bbox)
            dataset['valid_i'].append(valid_i)

            vid_segments.append(vid_uniq_id)
            vid_used_frames.append(img_file)
            vid_used_joints.append(joints_2d)
            vid_used_bbox.append(bbox)

        vid_segments = np.array(vid_segments)
        ids = np.zeros((len(set(vid_segments)) + 1))
        ids[-1] = len(vid_used_frames) + 1
        if (np.where(vid_segments[:-1] != vid_segments[1:])[0]).size != 0:
            ids[1:-1] = (np.where(vid_segments[:-1] != vid_segments[1:])[0]) + 1

        for i in tqdm(range(len(set(vid_segments)))):
            features = extract_features(model, None, np.array(vid_used_frames)[int(ids[i]):int(ids[i + 1])],
                                        vid_used_bbox[int(ids[i]):int(ids[i + 1])],
                                        kp_2d=np.array(vid_used_joints)[int(ids[i]):int(ids[i + 1])],
                                        dataset='spin', debug=False, scale=1.2)  # 1.0 for mpii3d_train_scale1_db.pt
            dataset['features'].append(features)

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
    dataset['features'] = np.concatenate(dataset['features'])

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default="/media/DATA2/wchuq/3DHPE/dataset/3dhp/")
    parser.add_argument('--ckpt', type=str, help='cliff checkpoint', default='data/base_data/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt')
    args = parser.parse_args()
    DB_DIR = "/media/DATA2/wchuq/3DHPE/TCMR_RELEASE-master/self_preprocess_data"
    # smpl_param = smpl_params[str(subject_idx)][str(seq_idx)][str(frame_idx)] 
    smpl_data = json.load(open("/media/DATA2/wchuq/3DHPE/dataset/3dhp/annot/MPI-INF-3DHP_SMPL_NeuralAnnot.json", 'r'))

    # smpl_data = json.load(open("/media/DATA2/wchuq/3DHPE/dataset/human3.6m/annotations/Human36M_subject1_smpl_param.json", 'r'))

    # dataset = read_test_data(args.dir)
    # joblib.dump(dataset, osp.join(DB_DIR, 'hr48_mpii3d_val_scale12_db.pt'))

    dataset = read_train_data(args.dir, smpl_data)
    joblib.dump(dataset, osp.join(DB_DIR, 'hr48_mpii3d_train_scale12_train_db.pt'))
    print("----------success save---------")



