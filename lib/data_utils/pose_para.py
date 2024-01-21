import sys


sys.path.append('.')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3' # 指定gpu
import cv2
import torch
import joblib
import argparse
import numpy as np
import pickle as pkl
import os.path as osp
from tqdm import tqdm
from kornia.geometry.conversions import angle_axis_to_rotation_matrix

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as transforms
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
# Although Axes3D is not used directly,
# it is imported because it is needed for 3d projection.

from robustgmm import RobustGMM


from lib.models import spin
# 使用resnet50或hr48
# from lib.backbone.cliff_res50 import CLIFF
from lib.backbone.cliff_hr48 import CLIFF
from lib.data_utils._kp_utils import *
from lib.core.config import TCMR_DB_DIR, BASE_DATA_DIR
from lib.utils.smooth_bbox import get_smooth_bbox_params, get_all_bbox_params
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14,SMPL_MEAN_PARAMS
from lib.utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis, matrix_to_rotation_6d


NUM_JOINTS = 24
VIS_THRESH = 0.3
MIN_KP = 6

'''
计算得到3DPW的pose参数分布
'''

# All functions are for visualization.
def make_ellipses(ax, means, covs, edgecolor, m_color, ls='-', n_std=3):
    def _make_ellipses(mean, cov):
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = mpl.patches.Ellipse((0, 0),
                                      ell_radius_x * 2,
                                      ell_radius_y * 2,
                                      facecolor='none',
                                      edgecolor=edgecolor, lw=1, ls=ls)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y)
        transf = transf.translate(mean[0], mean[1])
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)
        ax.add_artist(ellipse)
        ax.set_aspect('equal', 'datalim')
        ax.scatter(mean[0], mean[1], c=m_color, marker='*')
    for mean, cov in zip(means, covs):
        _make_ellipses(mean, cov)



if __name__ == '__main__':

    # para = joblib.load("self_preprocess_data/para/para.pt")
    
  

    db_file = "self_preprocess_data/para/hr48_3dpw_train_posepara_db.pt"


    dataset =  joblib.load(db_file)
    rotation_6d = dataset['rotation_6d'] # [batch, 144]
    rotation_angle = dataset['pose'] # [batch, 72]

    # GMM using Standard EM Algorithm with random initial values
    # 标准EM算法
    k = 5
    init_idx = np.random.choice(np.arange(rotation_angle.shape[0]), k)
    means_init = rotation_angle[init_idx, :]
    gmm = GaussianMixture(n_components=k, means_init=means_init)
    # https://scikit-learn.org/stable/modules/mixture.html#gmm
    # https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    gmm.fit(rotation_angle)
    para = {
        'mean': [],
        'covs': [],
        'weight': []

    }

    means_sklearn = gmm.means_ # 拟合后的gmm均值
    covs_sklearn = gmm.covariances_ # 多元高斯分布的协方差矩阵(k,feature,feature)
    weight_sklearn = gmm.weights_ # 权重
    para['mean'] = means_sklearn
    para['covs'] = covs_sklearn
    para['weight'] = weight_sklearn

    joblib.dump(para, osp.join("self_preprocess_data/para/", 'para_angle_axis_5.pt'))




    # # Visualization
    # plt.figure(figsize=(9, 4))
    # plt.subplots_adjust(wspace=.2)
    # ax1 = plt.subplot(1, 2, 1)
    # ax1.set_title('Real Data and Real Gaussian Distribution', fontsize=10)
    # ax1.scatter(rotation_6d[:, 0], rotation_6d[:, 1], marker='.', c='g', s=10)
    # ax1.set_xlim(np.min(rotation_6d[:, 0]) - 2, np.max(rotation_6d[:, 0]) +  2)
    # ax1.set_ylim(np.min(rotation_6d[:, 1]) -  2, np.max(rotation_6d[:, 1]) +  2)
    # ax2 = plt.subplot(1, 2, 2)
    # ax2.set_title('Standard EM with random initial values', fontsize=10)
    # ax2.scatter(rotation_6d[:, 0], rotation_6d[:, 1], marker='.', c='g', s=10)
    # ax2.set_xlim(np.min(rotation_6d[:, 0]) -  2, np.max(rotation_6d[:, 0]) +  2)
    # ax2.set_ylim(np.min(rotation_6d[:, 1]) -  2, np.max(rotation_6d[:, 1]) +  2)
    # make_ellipses(ax=ax2, means=means_sklearn, covs=covs_sklearn,
    #             edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
    # plt.suptitle('Example1')
    # plt.savefig('./self_preprocess_data/para/figure/train_10.png', dpi=300)


