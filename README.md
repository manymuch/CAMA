# CAMA
Official Implementation of A Vision-Centric Approach for Static Map Element Annotation  
[Arxiv](https://arxiv.org/abs/2309.11754) | [Youtube](https://www.youtube.com/watch?v=oBa4ngd2b9Y) | [Bilibili](https://www.bilibili.com/video/BV1ek4y1F7nJ)  

**CAMA**: **C**onsistent and **A**ccurate **M**ap **A**nnotation, nuScenes example:  
<p align="left">
  <img src="assets/cover.jpeg" width="70%"/>
</p>

CAMA is also used for detecting drowsiness driving patterns based on static map element matching. For instance, our proposed driving behavior dataset CAMA-D. Details here (https://github.com/FatigueView/fatigueview)
## Pipeline
<p align="left">
  <img src="assets/pipeline.png" width="70%"/>
</p>

## Release Notes  
### 1.0.0 (2023-10-13)  
* Upload nuScenes xxx scenes from v1.0-test with CAMA labels.  
* Add reprojection demo for both CAMA and nuScenes origin labels.  

## Run: Reprojection Demo  

1. install required python packages  
    ```bash
    python3 -m pip install -r requirements.txt  
    ```
2. Download cama_label.zip [[Google Drive](https://drive.google.com/file/d/1QUae0pMtXxfGCzjprN1_cKuXdjD854Qj/view?usp=sharing)]  

3. Modify config.yaml accordingly:  
    * **dataroot**: path to the origin nuScenes dataset  
    * **converted_dataroot**: output converted dataset dir  
    * **cama_label_file**: path to cama_label.zip you just download from 2  
    * **output_video_dir**: where the demo video writes
4. Run the pipeline  
    ```bash
    python3 main.py --config config.yaml
    ```

## Citation

If you benefit from this work, please cite the mentioned and our paper:

    @inproceedings{zhang2021deep,
      title={A Vision-Centric Approach for Static Map Element Annotation},
      author={Zhang, Jiaxin and Chen, Shiyuan and Yin, Haoran and Mei, Ruohong and Liu, Xuan and Yang, Cong and Zhang, Qian and Sui, Wei},
      booktitle={IEEE International Conference on Robotics and Automation (ICRA 2024)},
      pages={1-7}
    }

