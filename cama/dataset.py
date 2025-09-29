import cv2
import numpy as np
from os.path import join, exists

from tqdm import tqdm
from cama.dataset_reader import DatasetReader
from cama.pose_transformer import PoseTransformer
from cama.reproject import MapManager, CameraManager
from cama.tools import load_json
from cama.metric import MetricEvaluation


class ClipManager:
    def __init__(self, configs, clip_path=None, pose_prefix=None):
        self.configs = configs
        self.mm = MapManager(pose_prefix)
        self.instance_maps = dict()
        self.pose_prefix = pose_prefix
        if clip_path is not None:
            self.clip_path = clip_path
            self.cm_list = self.prepare_camera_manager(clip_path)
            cama_instance = self.load_clip_cama(clip_path)
            if cama_instance is not None:
                self.instance_maps["cama"] = cama_instance
            nuscenes_instance = self.load_clip_nuscenes(clip_path)
            if nuscenes_instance is not None:
                self.instance_maps["nuscenes"] = nuscenes_instance

    def load_clip_cama(self, clip_path):
        # load labels.json
        label_json = join(
            clip_path, self.configs["result_dir"], self.configs["cama_map_file"])
        if not exists(label_json):
            return None
        labels = load_json(label_json)

        # get 3D maps
        height_npy_path = join(
            clip_path, self.configs["result_dir"], self.configs["height_mlp"])
        bev_height = np.load(height_npy_path)

        instance_map = self.mm.calculate_3d_instance_maps(
            bev_height, labels)
        return instance_map

    def load_clip_nuscenes(self, clip_path):
        # load labels.json
        label_json = join(
            clip_path, self.configs["result_dir"], self.configs["nuscenes_map_file"])
        if not exists(label_json):
            return None
        labels = load_json(label_json)
        instance_map = self.mm.load_3d_instance_maps(labels)
        return instance_map

    def prepare_camera_manager(self, clip_path):
        cm_list = []
        for camera_name in self.configs["camera_list"]:
            cm = CameraManager(clip_path, camera_name)
            cm_list.append(cm)
        return cm_list

    def get_pt_cama(self, dr):
        camera_main = self.configs["camera_main"]
        chassis2camera_main = dr.get_extrinsic("chassis", camera_main)
        camera_pose = dr.get_odometry(f"{self.pose_prefix}_{camera_main}.txt")
        pt = PoseTransformer()
        pt.loadarray(camera_pose)
        pt.right_rotate(chassis2camera_main)
        # chassis2world
        return pt

    def get_pt_nuscenes(self, dr):
        camera_pose = dr.get_odometry("wigo_offset_clip.txt")
        pt = PoseTransformer()
        pt.loadarray(camera_pose)
        pt.normalize2center()
        return pt

    def yield_frame(self, dataset):
        camera_main = self.configs["camera_main"]
        # get chassis pose
        dr = DatasetReader(self.clip_path)
        if dataset == "nuscenes":
            pt = self.get_pt_nuscenes(dr)
        elif dataset == "cama":
            pt = self.get_pt_cama(dr)
        # iterate over each frame
        sensor_time_seconds = dr.get_sensor_timestamp(camera_main, sync=True)
        for image_idx in tqdm(range(1, len(sensor_time_seconds))):
            timestamp = sensor_time_seconds[image_idx]
            try:
                chassis2world = pt.seek_by_timestamp(
                    timestamp, t_max_diff=0.5, interpolate=True).astype(np.float32)
            except RuntimeError:
                # skip as there is no exact match time pose
                # print(image_idx)
                continue

            # transform instance map from world to chassis
            world2chassis = np.linalg.inv(chassis2world)
            instance_map = self.instance_maps[dataset]
            instance_map = self.mm.transform_3d_instance_maps(
                instance_map, world2chassis)

            # crop instance map
            instance_map = self.mm.crop_3d_instance_maps(instance_map)
            yield (image_idx, instance_map)

    def project_all_camera(self, maps_3d):
        maps_2d_dict = {}
        for cm in self.cm_list:
            chassis2camera = cm.get_chassis2camera()
            instance_map_camera = self.mm.transform_3d_instance_maps(
                maps_3d, chassis2camera)
            # project instance map to image
            maps_2d = cm.project_to_image(instance_map_camera)
            maps_2d_dict[cm.camera_name] = maps_2d
        return maps_2d_dict

    def render_vectors(self, maps_2d_dict, image_idx):
        render_image_dict = {}
        for cm in self.cm_list:
            image = cm.read_resized_image_by_index(image_idx)
            maps_2d = maps_2d_dict[cm.camera_name]
            render_image = cm.render_maps(image, maps_2d)
            render_image_dict[cm.camera_name] = render_image
        return render_image_dict

    def render_sres(self, image_dict, sres_image_dict):
        for camera_name, image in image_dict.items():
            linestring_list_list = sres_image_dict[camera_name]
            if linestring_list_list is None:
                continue
            for linestring_list in linestring_list_list:
                for linestring in linestring_list:
                    # plot linestring on image using cv2.line
                    start_point = (int(linestring.coords[0][1]), int(linestring.coords[0][0]))
                    end_point = (int(linestring.coords[-1][1]), int(linestring.coords[-1][0]))
                    image = cv2.line(image, start_point, end_point, (0, 0, 255), 1)
        return image_dict

    def gather_errors(self, errors_dict):
        error_dict = {}
        for camera_name, errors in errors_dict.items():
            if errors is None:
                continue
            mean_error = np.mean(errors)
            error_dict[camera_name] = mean_error
        return error_dict

    def render_instance(self, image_dict, instances_dict):
        for camera_name, image in image_dict.items():
            instances = instances_dict[camera_name]
            for semantic_id, masks in instances.items():
                for mask in masks:
                    image[mask] = [230, 170, 143]
        return image_dict

    def evaluate_all_camera(self, maps_2d_dict, image_idx):
        evaluate_dict = {}
        errors_dict = {}
        instance_dict = {}
        precisions_dict = {}
        recalls_dict = {}
        me = MetricEvaluation()

        # iterate over camera
        for cm in self.cm_list:

            # read 2D instance segmentation
            instances_png = cm.read_resized_instance_by_index(image_idx)
            instances_mask = me.load_instance(instances_png)
            vector_image = me.masks2skeleton(instances_mask)

            # get projected vecotrs
            vector_projected = maps_2d_dict[cm.camera_name]
            vector_projected = me.vectors_float2int(vector_projected)

            sres_vis, errors, precision, recall = me.calculate_sre(vector_image, vector_projected)
            instance_dict[cm.camera_name] = instances_mask
            evaluate_dict[cm.camera_name] = sres_vis
            errors_dict[cm.camera_name] = errors
            precisions_dict[cm.camera_name] = precision
            recalls_dict[cm.camera_name] = recall
        return evaluate_dict, errors_dict, instance_dict, precisions_dict, recalls_dict
