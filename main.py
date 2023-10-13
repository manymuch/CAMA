import os
import yaml
import zipfile
import argparse
from dataset.nuscenes2clip import nuScenes2Clip
from cama.tools import VideoGenerator
from cama.dataset import ClipManager


def extract_dir_from_zip(zip_filepath, dir_in_zip, dest_dir):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if member.startswith(dir_in_zip):
                zip_ref.extract(member, dest_dir)
                extracted_path = os.path.join(dest_dir, member)
                if member.endswith('/'):
                    os.makedirs(extracted_path, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a configuration file.')
    parser.add_argument('-c', '--config', type=str, default="config.yaml", help='Path to the configuration file.')
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    s2c = nuScenes2Clip(configs)
    output_dir = configs["converted_dataroot"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for scene_name in configs["scene_names"]:
        ##########################################################
        # Step1. convert nuScenes scene data to CAMA clip format #
        ##########################################################
        s2c.convert(scene_name)

        ##########################################################
        # Step2. copy cama label files into CAMA clip dir        #
        ##########################################################
        zip_file = configs["cama_label_file"]
        dir_in_zip = f"{scene_name}/"
        extract_dir_from_zip(zip_file, dir_in_zip, output_dir)

        ##########################################################
        # Step3. make reprojection video                         #
        ##########################################################
        output_video_dir = configs["output_video_dir"]
        clip_path = os.path.join(output_dir, scene_name)
        cm = ClipManager(configs["cama_configs"], clip_path)
        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir)

        # cama reproject
        print("Generating reprojection video with CAMA labels...")
        vg = VideoGenerator(os.path.join(output_video_dir, f"{scene_name}_cama.mp4"))
        for image_idx, instance_map in cm.yield_frame(dataset="cama"):
            maps_2d_dict = cm.project_all_camera(instance_map)
            image_dict = cm.render_vectors(maps_2d_dict, image_idx)
            image = vg.concate_image(image_dict)
            vg.add_frame(image)

        # nuscenes reproject
        print("Generating reprojection video with nuScenes labels...")
        vg = VideoGenerator(os.path.join(output_video_dir, f"{scene_name}_nuScenes.mp4"))
        for image_idx, instance_map in cm.yield_frame(dataset="nuscenes"):
            maps_2d_dict = cm.project_all_camera(instance_map)
            image_dict = cm.render_vectors(maps_2d_dict, image_idx)
            image = vg.concate_image(image_dict)
            vg.add_frame(image)
