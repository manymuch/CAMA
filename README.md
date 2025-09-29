# CAMA
Official Implementation of A Vision-Centric Approach for Static Map Element Annotation  
CAMA： [Arxiv](https://arxiv.org/abs/2309.11754) | [Youtube](https://www.youtube.com/watch?v=oBa4ngd2b9Y) | [Bilibili](https://www.bilibili.com/video/BV1ek4y1F7nJ)  
CAMAv2： [Arxiv](https://arxiv.org/abs/2407.21331) | [Youtube](https://www.youtube.com/watch?v=npbbOEpuTno) | [Bilibili](https://www.bilibili.com/video/BV1afeFeAEsg)

**CAMA**: **C**onsistent and **A**ccurate **M**ap **A**nnotation, nuScenes example:  
<p align="left">
  <img src="assets/cover.jpeg" width="70%"/>
</p>

## Pipeline
<p align="left">
  <img src="assets/pipeline.jpg" width="70%"/>
</p>

## Release Notes
Please run ```git checkout camav2``` to switch to camav2 branch.
### 2.1.0 (2025-09-29)
* Release the evaluation scripts (SRE, precision, recall, F1-score).
* Add LiDAR aggregation demo using CAMAv2 reconstructed poses.
### 2.0.0 (2024-07-31)
* camav2_label.zip [[Google Drive](https://drive.google.com/file/d/1zRZIB7BHKS_6sbC8oBA1mFe9_eaTMPs5/view?usp=sharing)]
* CAMAv2 aggregates scenes with intersecting portions into one large scene called a **site**.
* It solves the shortcoming of dropping head and tail frames in camav1.
### 1.0.0 (2023-10-13)  
* cama_label.zip [[Google Drive](https://drive.google.com/file/d/1QUae0pMtXxfGCzjprN1_cKuXdjD854Qj/view?usp=sharing)]  
* Upload nuScenes 73 scenes from v1.0-test with CAMA labels.  
* Add reprojection demo for both CAMA and nuScenes origin labels.  

## Prepare camav2 labels
Download the [camav2_label.zip](https://drive.google.com/file/d/1zRZIB7BHKS_6sbC8oBA1mFe9_eaTMPs5/view?usp=sharing)

```bash
unzip camav2_label.zip

cd camav2_label

# create symlinks for 'site' reconstruction clips to the site2clip folder
python site2clip.py
```

### Folder structure
There are two kinds of labels
* Clip mode (scmv): ```map_height = map_width = 300```
* Site mode (mcmv): ```map_heigth = map_width = 600```

The data is organized in the following format:

```
/camav2_labels/
          |-- all_clips/
          │       |-- scene-0001/
          │       |   |-- maps/
          │       |         |- map_labels.json
          │       |         |- vision_road_mlp_ft.npy
          │       |-- |-- odometry/                          # site mode
          │       |         |- mcmv_camera_front_left.txt
          │       |         |- mcmv_camera_front_right.txt
          │       |         |- mcmv_camera_front.txt
          │       |         |- mcmv_camera_rear_left.txt
          │       |         |- mcmv_camera_rear_right.txt
          │       |         |- mcmv_camera_rear.txt
          │       |-- ...
          │       |-- scene-1101/
          │       |   |-- maps/
          │       |         |- map_labels.json
          │       |         |- vision_road_mlp_ft.npy
          │       |-- |-- odometry/                          # clip mode
          │       |         |- scmv_camera_front_left.txt
          │       |         |- scmv_camera_front_right.txt
          │       |         |- scmv_camera_front.txt
          │       |         |- scmv_camera_rear_left.txt
          │       |         |- scmv_camera_rear_right.txt
          │       |         |- scmv_camera_rear.txt
          |-- site2clip/
          │       |   |-- bs_site01_map_labels.json
          │       |   |-- bs_site01_vision_road_mlp_ft.npy
          |       |   |-- ...
          |-- site2clip.json
          |-- site2clip.py
```

## Run: Reprojection Demo  

1. install required python packages  
    ```bash
    python3 -m pip install -r requirements.txt  
    ```
2. Modify config.yaml accordingly:  
    * **dataroot**: path to the origin nuScenes dataset  
    * **converted_dataroot**: path to camav2_labels directory 
    * **output_video_dir**: where the demo video writes
    * **compute_metrics**: whether compute the evalutation metrics. 
    
3. Run the pipeline  
    ```bash
    python3 main.py --config config.yaml
    ```

 _Note: To compute evaluation metrics, first generate the lane instance segmentation for each camera and place the results under ```scene-xxxx/lane_ins_{cam_name}```. Sample data is provided [here](https://drive.google.com/file/d/1IdnYEhDg5fmrgmZCAPzOSuI-BxJMQGtH/view?usp=sharing)._

## Citation

If you benefit from this work, please cite the mentioned and our paper:

    @inproceedings{zhang2021deep,
      title={A Vision-Centric Approach for Static Map Element Annotation},
      author={Zhang, Jiaxin and Chen, Shiyuan and Yin, Haoran and Mei, Ruohong and Liu, Xuan and Yang, Cong and Zhang, Qian and Sui, Wei},
      booktitle={IEEE International Conference on Robotics and Automation (ICRA 2024)},
      pages={1-7}
    }

    @article{chen2024camav2,
      title={CAMAv2: A Vision-Centric Approach for Static Map Element Annotation},
      author={Chen, Shiyuan and Zhang, Jiaxin and Mei, Ruohong and Cai, Yingfeng and Yin, Haoran and Chen, Tao and Sui, Wei and Yang, Cong},
      journal={arXiv preprint arXiv:2407.21331},
      year={2024}
    }
  

