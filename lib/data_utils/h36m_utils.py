import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 指定gpu
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
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14,SMPL_MEAN_PARAMS
from lib.backbone.cliff_hr48 import CLIFF
from lib.utils.utils import tqdm_enumerate,strip_prefix_if_present
from lib.data_utils._kp_utils import convert_kps
from lib.data_utils._feature_extractor import extract_features


from lib.utils.smooth_bbox import get_smooth_bbox_params, get_all_bbox_params
from lib.utils.vis import draw_skeleton


VIS_THRESH = 0.3


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:,0] - c[0]) / f[0] * pixel_coord[:,2]
    y = (pixel_coord[:,1] - c[1]) / f[1] * pixel_coord[:,2]
    z = pixel_coord[:,2]
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1,3)).transpose(1,0)).transpose(1,0)
    return world_coord


def read_data_train(dataset_path, set='train', debug=False):
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

    # occluders = load_occluders("/media/DATA2/wchuq/3DHPE/dataset/voc/VOCdevkit/VOC2012/")

    # model = spin.get_pretrained_hmr()

    import torch

    # load model
    device = 'cuda'
    model = CLIFF(SMPL_MEAN_PARAMS).to(device)
    print("Load the CLIFF checkpoint from path:", args.ckpt)
    state_dict = torch.load(args.ckpt)['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    # model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.load_state_dict(state_dict, strict=False)

    if set == 'train':
        subjects = [1,5,6,7,8]
    else:
        subjects= [9,11]
    for subject in subjects:
        annot_path = osp.join(dataset_path, 'annotations')
        # camera load
        with open(osp.join(annot_path, 'Human36M_subject' + str(subject) + '_camera.json'), 'r') as f:
            cameras = json.load(f)
        # joint coordinate load
        with open(osp.join(annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
            joints = json.load(f)
        # SMPL parameters obtained by NeuralAnnot will be released (https://arxiv.org/abs/2011.11232) after publication
        ###注释部分
        # smpl parameter load
        with open(osp.join(annot_path, 'Human36M_subject' + str(subject) + '_smpl_param.json'), 'r') as f:
            smpl_params = json.load(f)
        ####

        seq_list = sorted (glob.glob(f"{dataset_path}/images/s_{subject:02d}*"))
        for seq in tqdm(seq_list):
            seq_name = seq.split('/')[-1]
            act = str(int(seq_name.split('_act_')[-1][0:2]))
            subact = str(int(seq_name.split('_subact_')[-1][0:2]))
            cam = str(int(seq_name.split('_ca_')[-1][0:2]))
            # if cam != '4':  # front camera (Table 6)
            #     continue
            print("seq name: ", seq)

            img_paths = sorted(glob.glob(seq + '/*.jpg'))
            # num_frames = len(img_paths)
            num_frames = len(joints[act][subact]) # 有多出标签的图片
            if num_frames < 1:
                continue
            # camera parameter
            cam_param = cameras[cam]
            R, t, f, c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(
                cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)

            # img starts from index 1, and annot starts from index 0
            poses = np.zeros((num_frames, 72), dtype=np.float32)
            shapes = np.zeros((num_frames, 10), dtype=np.float32)
            j3ds = np.zeros((num_frames, 49, 3), dtype=np.float32)
            j2ds = np.zeros((num_frames, 49, 3), dtype=np.float32)

            for img_i in tqdm(range(num_frames)):
                if img_i % 5 == 0:\
                    # smpl_param = smpl_params[act][subact][str(img_i)][cam]
                    try:
                        smpl_param = smpl_params[act][subact][str(img_i)]
                        pose = np.array(smpl_param['pose'], dtype=np.float32)
                        shape = np.array(smpl_param['shape'], dtype=np.float32)
                    except KeyError:
                        pose = np.array(np.zeros(72), dtype=np.float32)
                        shape = np.array(np.zeros(10), dtype=np.float32)
                        print("data missed")
                    ####


                    joint_world = np.array(joints[act][subact][str(img_i)], dtype=np.float32)
                    # match right, left
                    match = [[1, 4], [2, 5], [3, 6]]
                    for m in match:
                        l, r = m
                        joint_world[l], joint_world[r] = joint_world[r].copy(), joint_world[l].copy()
                    joint_cam = world2cam(joint_world, R, t)
                    joint_img = cam2pixel(joint_cam, f, c)

                    j3d = convert_kps(joint_cam[None, :, :] / 1000, "h36m", "spin").reshape((-1, 3))
                    j3d = j3d - j3d[39]  # 4 is the root

                    joint_img[:, 2] = 1
                    j2d = convert_kps(joint_img[None, :, :], "h36m", "spin").reshape((-1,3))

                    poses[img_i] = pose
                    shapes[img_i] = shape
                    j3ds[img_i] = j3d
                    j2ds[img_i] = j2d



            bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2ds, vis_thresh=VIS_THRESH, sigma=8)
            # bbox_params, time_pt1, time_pt2 = get_all_bbox_params(j2ds, vis_thresh=VIS_THRESH)

            # img = cv2.imread(img_paths[0])
            # temp = draw_skeleton(img, j2ds[0], dataset='spin', unnormalize=False, thickness=2)
            # cv2.imshow('img', temp)
            # cv2.imwrite('./demo_show/img.jpg',temp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.waitKey(1)
            

            # process bbox_params
            c_x = bbox_params[:, 0]
            c_y = bbox_params[:, 1]
            scale = bbox_params[:, 2]

            w = h = 150. / scale
            # w = h = h * 0.9  # 1.1 for h36m_train_25fps_occ_db.pt
            w = h = h * 1.1  # 1.1 for h36m_train_25fps_occ_db.pt
            bbox = np.vstack([c_x, c_y, w, h]).T

            img_paths_array = np.array(img_paths)[time_pt1:time_pt2][::5]
            bbox = bbox[::5]
            # subsample frame to 25 fps

            dataset['vid_name'].append(np.array([f'{seq}_{subject}'] * num_frames)[time_pt1:time_pt2][::5])
            dataset['frame_id'].append(np.arange(0, num_frames)[time_pt1:time_pt2][::5])
            dataset['joints3D'].append(j3ds[time_pt1:time_pt2][::5])
            dataset['joints2D'].append(j2ds[time_pt1:time_pt2][::5])
            dataset['shape'].append(shapes[time_pt1:time_pt2][::5])
            dataset['pose'].append(poses[time_pt1:time_pt2][::5])

            dataset['img_name'].append(img_paths_array)
            dataset['bbox'].append(bbox)

            features = extract_features(model, None, img_paths_array, bbox,
                                        kp_2d=j2ds[time_pt1:time_pt2][::5], debug=debug, dataset='h36m', scale=0.9)  # 1.2 for h36m_train_25fps_occ_db.pt

            dataset['features'].append(features)

    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])
        print(k, dataset[k].shape)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default="/media/DATA2/wchuq/3DHPE/dataset/human3.6m/")
    parser.add_argument('--set', type=str, help='select train/test set', default='train')
    parser.add_argument('--ckpt', type=str, help='cliff checkpoint', default='data/base_data/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt')
    args = parser.parse_args()

    # import torch
    torch.set_num_threads(8)


    dataset = read_data_train(args.dir, args.set)
    joblib.dump(dataset, osp.join("self_preprocess_data/h36m", f'hr48_h36m_{args.set}_db.pt'))  # h36m_train_25fps_occ_db.pt



