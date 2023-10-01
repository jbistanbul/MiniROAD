# MiniROAD: Minimal RNN Framework for Online Action Detection
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/miniroad-minimal-rnn-framework-for-online/online-action-detection-on-fineaction)](https://paperswithcode.com/sota/online-action-detection-on-fineaction?p=miniroad-minimal-rnn-framework-for-online)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/miniroad-minimal-rnn-framework-for-online/online-action-detection-on-thumos-14)](https://paperswithcode.com/sota/online-action-detection-on-thumos-14?p=miniroad-minimal-rnn-framework-for-online)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/miniroad-minimal-rnn-framework-for-online/online-action-detection-on-tvseries)](https://paperswithcode.com/sota/online-action-detection-on-tvseries?p=miniroad-minimal-rnn-framework-for-online)

## Introduction

This is a pytorch implementation for our ICCV 2023 paper "[`MiniROAD: Minimal RNN Framework for Online Action Detection`]()".

![teaser](assets/miniroad_teaser.png?raw=true)

## Data Preparation

#### THUMOS14 and TVSeries

To prepare the features and targets by yourself, please refer to [LSTR](https://github.com/amazon-research/long-short-term-transformer#data-preparation). You can also directly download the pre-extracted features and targets from [`TeSTra`](https://github.com/zhaoyue-zephyrus/TeSTra).

#### FineAction

Download the officially available pre-extracted features from [`FineAction`](https://github.com/Richard-61/FineAction). As mentioned in the paper, the temporal dimensions have been linearly interpolated by a factor of four as the officially available feature is too condensed (16 frames being converted into one feature).

#### Data Structure

1. If you want to use our [dataloaders](datasets/dataset.py), please make sure to put the files as the following structure:

    * THUMOS'14 dataset:
        ```
        $YOUR_PATH_TO_THUMOS_DATASET
        ├── rgb_FEATURETYPE/
        |   ├── video_validation_0000051.npy 
        │   ├── ...
        ├── flow_FEATURETYPE/ 
        |   ├── video_validation_0000051.npy 
        |   ├── ...
        ├── target_perframe/
        |   ├── video_validation_0000051.npy (of size L x 22)
        |   ├── ...
        ```
    
    * TVSeries dataset:
        ```
        $YOUR_PATH_TO_TVSERIES_DATASET
        ├── rgb_FEATURETYPE/
        |   ├── Breaking_Bad_ep1.npy 
        │   ├── ...
        ├── flow_FEATURETYPE/
        |   ├── Breaking_Bad_ep1.npy 
        |   ├── ...
        ├── target_perframe/
        |   ├── Breaking_Bad_ep1.npy (of size L x 31)
        |   ├── ...
        ```

    * FineAction dataset:
        ```
        $YOUR_PATH_TO_FINEACTION_DATASET
        ├── rgb_kinetics_i3d/
        |   ├── v_00008645.npy (of size L x 2048)
        │   ├── ...
        ├── flow_kinetics_i3d/
        |   ├── v_00008645.npy (of size L x 2048)
        |   ├── ...
        ├── target_perframe/
        |   ├── v_00008645.npy (of size L x 107)
        |   ├── ...
        ```
    For appropriate FEATURETYPE, please refer to (datasets/dataset.py)

2. Create softlinks of datasets:

    ```
    cd MiniROAD
    ln -s $YOUR_PATH_TO_THUMOS_DATASET data/THUMOS
    ln -s $YOUR_PATH_TO_TVSERIES_DATASET data/TVSERIES
    ln -s $YOUR_PATH_TO_FINEACTION_DATASET data/FINEACTION
    ```

## Training

    ```
    cd MiniROAD
    python main.py --config $PATH_TO_CONFIG_FILE 
    ```

## Inference from checkpoint

    ```
    cd MiniROAD
    python main.py --config $PATH_TO_CONFIG_FILE --eval $PATH_TO_CHECKPOINT
    ```

## Main Results and checkpoints

### THUMOS14

|       method      | feature   |  mAP (%)  |                             config                                                |   checkpoint   |
|  :--------------: |  :-------------:  |  :-----:  |  :-----------------------------------------------------------------------------:  |  :----------:  |
|  MiniROAD           |  kinetics |   71.8    | [yaml](configs/miniroad_thumos_kinetics.yaml) | [Download](https://yonsei-my.sharepoint.com/:u:/g/personal/jbistanbul05_o365_yonsei_ac_kr/EQ1XQBlxzgZChRV1faSQKTwBEGIcZILS8mj1QPTzAVva9Q?e=G5YfWP) |
|  MiniROAD           |    nv_kinetics    |   68.4    | [yaml](configs/miniroad_thumos_nv_kinetics.yaml)      | [Download](https://yonsei-my.sharepoint.com/:u:/g/personal/jbistanbul05_o365_yonsei_ac_kr/EdxCH-yklvJAllMzjMburMEBBz0ULEPMf16hKRlvAA_4kg?e=0jYLzR) |

### FINEACTION

|       method      | feature   |  mAP (%)  |                             config                                                |   checkpoint   |
|  :--------------: |  :-------------:  |  :-----:  |  :-----------------------------------------------------------------------------:  |  :----------:  |
|  MiniROAD           |  kinetics |   37.1    | [yaml](configs/miniroad_fineaction_kinetics.yaml) | [Download](https://yonsei-my.sharepoint.com/:u:/g/personal/jbistanbul05_o365_yonsei_ac_kr/Eeldj4uYggxFlYzYY3J-7j4B6rqr1uA2PbsF18G_97w1ew?e=DKwwd3) |

### TVSERIES

|       method      | feature   |  mcAP (%)  |                             config                                                |   checkpoint   |
|  :--------------: |  :-------------:  |  :-----:  |  :-----------------------------------------------------------------------------:  |  :----------:  |
|  MiniROAD           |  kinetics |   89.6    | [yaml](configs/miniroad_tvseries_kinetics.yaml) | [Download](https://yonsei-my.sharepoint.com/:u:/g/personal/jbistanbul05_o365_yonsei_ac_kr/ES4wVDcYIZNNmyzupS2UUoABeo6i7oFvUnP7HpdQ7y1b8g?e=PvISCv) |

## Citations

If you are using the data/code/model provided here in a publication, please cite our paper:

```BibTeX
@inproceedings{miniroad,
	title={MiniROAD: Minimal RNN Framework for Online Action Detection},
	author={An, Joungbin and Kang, Hyolim and Han, Su Ho and Yang, Ming-Hsuan and Kim, Seon Joo},
	booktitle={International Conference on Computer Vision (ICCV)},
	year={2023}
}
```

## License

This project is licensed under the Apache-2.0 License.

## Acknowledgements

Many of the codebase is from [LSTR](https://github.com/amazon-research/long-short-term-transformer).
