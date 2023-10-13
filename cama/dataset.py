import numpy as np
from os.path import join, exists

from tqdm import tqdm
from horizon_driving_dataset import DatasetReader, PoseTransformer
from cama.reproject import MapManager, CameraManager
from cama.tools import load_json


class ClipManager:
    def __init__(self, configs, clip_path=None):
        self.configs = configs
        self.mm = MapManager()
        self.instance_maps = dict()
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
        pose_prefix = self.configs["pose_prefix"]
        camera_pose = dr.get_odometry(f"{pose_prefix}_{camera_main}.txt")
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
