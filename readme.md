

We provide hrnet-based data preprocessing in Diff-HMR, and most of the scripts follow the[ TCMR](https://github.com/hongsukchoi/TCMR_RELEASE/blob/master/asset/data.md). Download pre-processed data in this [link](https://drive.google.com/drive/folders/16o4OE8DZLWL8Mr1Oy066dsucTSoNEdjI). (Full data will be uploaded soon)

You may also download datasets from source and pre-process yourself, following the guidelines below.

# Data Pre-Processing

## 1. Download Datasets

You should first down load the datasets used in Diff-HMR.

- **[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW)**

Directory structure:

```shell
3dpw
|-- imageFiles
|   |-- courtyard_arguing_00
|   |-- courtyard_backpack_00
|   |-- ...
`-- sequenceFiles
    |-- test
    |-- train
    `-- validation
```

- **[Human 3.6M](http://vision.imar.ro/human3.6m/description.php)**

Once getting available to the Human 3.6M dataset, one could refer to [the script](https://github.com/nkolot/SPIN/blob/master/datasets/preprocess/h36m_train.py) from the official SPIN repository to preprocess the Human 3.6M dataset.
Directory structure:

```shell
human3.6m
|-- annot
|-- dataset_extras
|-- S1
|-- S11
|-- S5
|-- S6
|-- S7
|-- S8
`-- S9
```

- **InstaVariety**

Download the
[preprocessed tfrecords](https://github.com/akanazawa/human_dynamics/blob/master/doc/insta_variety.md#pre-processed-tfrecords) 
provided by the authors of Temporal HMR.

Directory structure:

```shell
insta_variety
|-- train
|   |-- insta_variety_00_copy00_hmr_noS5.ckpt-642561.tfrecord
|   |-- insta_variety_01_copy00_hmr_noS5.ckpt-642561.tfrecord
|   `-- ...
`-- test
    |-- insta_variety_00_copy00_hmr_noS5.ckpt-642561.tfrecord
    |-- insta_variety_01_copy00_hmr_noS5.ckpt-642561.tfrecord
    `-- ...
```



- **[PoseTrack](https://posetrack.net/)** 

Directory structure: 

```shell
posetrack
|-- images
|   |-- train
|   |-- val
|   |-- test
`-- posetrack_data
    `-- annotations
        |-- train
        |-- val
        `-- test
```

## 2. Download HR-Net Checkpoint

 Download model checkpoint for HR-Net from this [link]([ckpt - Google 云端硬盘](https://drive.google.com/drive/folders/1dAZiPqJY2wBv6QzpjOwYi4Ax1y-oBIM1))

We load the pre-trained model and take the HR-Net output to save the features, details can be found in the code.

## 2. Run data-preprocessing scirpts

```
python lib/data_utils/{dataset_name}_utils.py --dir ./data/{dataset_name} --ckpt ./data/{hr_net_ckpt}
```

## It's Done!

After downloading all the datasets and pre-process, all the data has been done.
