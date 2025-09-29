import os
import yaml
import zipfile
import argparse
import numpy as np
import glob
from dataset.nuscenes2clip import nuScenes2Clip
from cama.tools import VideoGenerator
from cama.dataset import ClipManager


def detect_pose_prefix(pose_dir, prefer=("mcmv", "scmv")):
    for p in prefer:
        if glob.glob(os.path.join(pose_dir, f"{p}_*.txt")):
            return p
    return None

def extract_dir_from_zip(zip_filepath, dir_in_zip, dest_dir):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if member.startswith(dir_in_zip):
                zip_ref.extract(member, dest_dir)
                extracted_path = os.path.join(dest_dir, member)
                if member.endswith('/'):
                    os.makedirs(extracted_path, exist_ok=True)

def generate_reprojection_video(cm, output_video_dir, scene_name, dataset, compute_metrics=False):
    print(f"Generating reprojection video with {dataset} labels...")
    vg = VideoGenerator(os.path.join(output_video_dir, f"{scene_name}_{dataset}.mp4"))

    mean_error_list = []
    mean_precision_list = []
    mean_recall_list = []

    for image_idx, instance_map in cm.yield_frame(dataset=dataset):
        maps_2d_dict = cm.project_all_camera(instance_map)
        image_dict = cm.render_vectors(maps_2d_dict, image_idx)

        if compute_metrics:
            sres_dict, errors_dict, instance_dict, precision_dict, recall_dict = cm.evaluate_all_camera(
                maps_2d_dict, image_idx
            )

            error_dict = cm.gather_errors(errors_dict)
            error_list = [v for v in error_dict.values() if v is not None]
            errors = np.mean(error_list) if len(error_list) > 0 else None

            precision_list = [v for v in precision_dict.values() if v is not None]
            precision = np.mean(precision_list) if len(precision_list) > 0 else None

            recall_list = [v for v in recall_dict.values() if v is not None]
            recall = np.mean(recall_list) if len(recall_list) > 0 else None
            
            if (errors is not None) and (precision is not None) and (recall is not None):
                mean_error_list.append(errors)
                mean_precision_list.append(precision)
                mean_recall_list.append(recall)
        
        image = vg.concate_image(image_dict)
        vg.add_frame(image)

    metrics = None
    if compute_metrics:
        if len(mean_error_list) > 0:
            mean_error = np.mean(mean_error_list)
            mean_precision = np.mean(mean_precision_list)
            mean_recall = np.mean(mean_recall_list)
            denom = (mean_precision + mean_recall)
            f1_score = float(2 * mean_precision * mean_recall / denom) if denom > 0 else 0.0
            metrics = dict(
                mean_error=mean_error,
                mean_precision=mean_precision,
                mean_recall=mean_recall,
                f1=f1_score
            )
            print(f"SRE: {mean_error:.02f}, Precision: {mean_precision:.02f}, "
                  f"Recall: {mean_recall:.02f}, F1: {f1_score:.02f}")
        else:
            print("No valid metrics were accumulated for this run.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a configuration file.')
    parser.add_argument('-c', '--config', type=str, default="config.yaml", help='Path to the configuration file.')
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    compute_metrics = configs.get("compute_metrics", False)
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
        # Step2. make reprojection video                         #
        ##########################################################
        output_video_dir = configs["output_video_dir"]
        clip_path = os.path.join(output_dir, scene_name)
        
        pose_prefix = detect_pose_prefix(os.path.join(clip_path, "odometry"))
        if pose_prefix is None:
            raise RuntimeError("Cannot detect pose prefix in odometry folder.")
        
        cm = ClipManager(configs["cama_configs"], clip_path, pose_prefix=pose_prefix)
        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir)

        # cama reproject
        generate_reprojection_video(cm, output_video_dir, scene_name,
                                   dataset="cama", compute_metrics=compute_metrics)
        # nuscenes reproject
        generate_reprojection_video(cm, output_video_dir, scene_name,
                                    dataset="nuscenes", compute_metrics=compute_metrics)
