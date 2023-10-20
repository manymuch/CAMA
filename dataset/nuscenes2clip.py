
import os
import shutil
import json
import logging
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import box, MultiPolygon, MultiLineString
from shapely import affinity, ops

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


class VectorizedLocalMap(object):
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'others': -1
    }

    def __init__(self,
                 dataroot,
                 patch_size,
                 map_classes=['divider', 'ped_crossing', 'boundary'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_ptsnum_per_line=-1,
                 padding_value=-10000, ):
        '''
        Args:
            fixed_ptsnum_per_line = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = dataroot
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value

    def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation, patch_size, patch_center):
        '''
        use lidar2global to get gt map layers
        '''

        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        patch_box = (patch_center[0], patch_center[1], patch_size[0], patch_size[1])
        # patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(patch_box, map_pose, patch_angle, self.line_classes, location)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(patch_box, map_pose, patch_angle, self.ped_crossing_classes, location)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    vectors.append((instance, self.CLASS2LABEL.get('ped_crossing', -1)))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(patch_box, map_pose, patch_angle, self.polygon_classes, location)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for contour in poly_bound_list:
                    vectors.append((contour, self.CLASS2LABEL.get('contours', -1)))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')

        gt_labels = []
        gt_instance = []
        for instance, type in vectors:
            if type != -1:
                gt_instance.append(instance)
                gt_labels.append(type)

        anns_results = dict(
            gt_vecs_pts_loc=gt_instance,
            gt_vecs_label=gt_labels,

        )
        return anns_results

    def get_map_geom(self, patch_box, map_pose, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(patch_box, map_pose, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(patch_box, map_pose, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, map_pose, patch_angle, location)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def ped_poly_geoms_to_instances(self, ped_geom):
        ped = ped_geom[0][1]
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_instances = self._one_type_line_geom_to_instances(a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_contour_line(self, patch_box, map_pose, patch_angle, layer_name, location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        # patch_x = patch_box[0]
        # patch_y = patch_box[1]
        patch_x = map_pose[0]
        patch_y = map_pose[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle=0)

        records = getattr(self.map_explorer[location].map_api, layer_name)

        polygon_list = []
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_explorer[location].map_api.extract_polygon(polygon_token) for polygon_token in
                            record['polygon_tokens']]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_explorer[location].map_api.extract_polygon(record['polygon_token'])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list

    def get_divider_line(self, patch_box, map_pose, patch_angle, layer_name, location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == 'traffic_light':
            return None

        # patch_x = patch_box[0]
        # patch_y = patch_box[1]
        patch_x = map_pose[0]
        patch_y = map_pose[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle=0)

        line_list = []
        records = getattr(self.map_explorer[location].map_api, layer_name)
        for record in records:
            line = self.map_explorer[location].map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list

    def get_ped_crossing_line(self, patch_box, map_pose, patch_angle, location):
        # patch_x = patch_box[0]
        # patch_y = patch_box[1]
        patch_x = map_pose[0]
        patch_y = map_pose[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle=0)
        polygon_list = []
        records = getattr(self.map_explorer[location].map_api, 'ped_crossing')
        # records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].map_api.extract_polygon(record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                  origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1,
                                                                                                                   2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1,
                                                                                                                   2)

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

        return sampled_points, num_valid


class nuScenes2Clip:
    def __init__(self, configs):
        self.nusc = NuScenes(version=configs["version"], dataroot=configs["dataroot"], verbose=True)
        self.configs = configs

        # start loading all keyframes
        self.samples = [samp for samp in self.nusc.sample]

        # converted sensor name
        self.clip_sensor_names = [
            'camera_front', 'camera_front_right', 'camera_front_left',
            'camera_rear', 'camera_rear_left', 'camera_rear_right',
            'lidar_top'
        ]
        # nuscenes sensor name
        self.scene_sensor_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
            'LIDAR_TOP'
        ]

    def compute_extrinsic2chassis(self, samp):
        calibrated_sensor = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
        q = calibrated_sensor["rotation"]  # w x y z
        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        tran = np.expand_dims(np.array(calibrated_sensor["translation"]), axis=0)
        sensor2chassis = np.hstack((rot, tran.T))
        sensor2chassis = np.vstack((sensor2chassis, np.array([[0, 0, 0, 1]])))  # [4, 4] camera 3D
        return sensor2chassis

    def get_scene_by_name(self, scene_name):
        for scene in self.nusc.scene:
            if scene['name'] == scene_name:
                return scene
        return None

    def write_odometry(self, clip_root, sweeps_sd_tokens):
        all_frames = []
        for index, value in enumerate(self.clip_sensor_names):
            token_list = sweeps_sd_tokens[value]
            for token in token_list:
                all_frames.append(self.nusc.get('sample_data', token))
        all_frames.sort(key=lambda x: (x['timestamp']))

        logger.info("Writing odometry/wigo.txt")
        od_path = os.path.join(clip_root, 'odometry')
        if not os.path.exists(od_path):
            os.makedirs(od_path)
        tum_array_list = []
        for i in range(len(all_frames)):
            sd = all_frames[i]
            pose = self.nusc.get('ego_pose', sd['ego_pose_token'])
            ts = sd['timestamp'] / 1e6
            pose_r = pose['rotation']
            pose_t = pose['translation']
            txyzqxqyqzqw = [ts, pose_t[0], pose_t[1], pose_t[2], pose_r[1], pose_r[2], pose_r[3], pose_r[0]]
            tum_array_list.append(txyzqxqyqzqw)
        tum_array = np.array(tum_array_list)
        np.savetxt(os.path.join(od_path, 'wigo.txt'), tum_array)
        logger.info("Done odometry/wigo.txt")

        wigo_offset_file = os.path.join(od_path, "wigo_offset_clip.txt")
        utm_center = deepcopy(tum_array[int(len(tum_array)/2), 1:4])
        tum_array[:, 1:4] = tum_array[:, 1:4] - deepcopy(utm_center)
        np.savetxt(wigo_offset_file, tum_array)
        logger.info("Done odometry/wigo_offset_clip.txt")

    def get_calibration(self, records):
        calibration = dict()
        record = records[0]
        camera_types = self.scene_sensor_names[:-1]
        for cam_index, cam in enumerate(camera_types):
            cam_token = record['data'][cam]
            sd_cam = self.nusc.get('sample_data', cam_token)
            # cam2chassis
            camera2chassis = self.compute_extrinsic2chassis(sd_cam)
            _, _, cam_intrinsic = self.nusc.get_sample_data(cam_token)
            calibration.update({self.clip_sensor_names[cam_index] + '_2_chassis': camera2chassis.tolist()})
            fov = 110 if cam == 'CAM_BACK' else 70
            cam_xxx_intr = {
                "center_u": cam_intrinsic[0, 2],
                "center_v": cam_intrinsic[1, 2],
                "distort": [0, 0, 0, 0, 0, 0, 0, 0],
                "focal_u": cam_intrinsic[0, 0],
                "focal_v": cam_intrinsic[1, 1],
                "fov": fov,
                "image_height": 900,
                "image_width": 1600,
                "K": cam_intrinsic.tolist(),
                "d": [0, 0, 0, 0, 0, 0, 0, 0]
            }
            calibration.update({self.clip_sensor_names[cam_index]: cam_xxx_intr})

        rec = records[0]
        lidar_token = rec['data']['LIDAR_TOP']
        sd_rec = self.nusc.get('sample_data', lidar_token)
        lidar2chassis = self.compute_extrinsic2chassis(sd_rec)
        calibration.update({'lidar_top_2_chassis': lidar2chassis.tolist()})
        return calibration

    def write_sensors(self, sweeps_sd_tokens, clip_root):
        unsync = dict()
        # prepare output dir
        for sensor_name in self.clip_sensor_names:
            view_path = os.path.join(clip_root, sensor_name)
            if not os.path.exists(view_path):
                os.makedirs(view_path)
        # iterate over each sensor
        for index, sensor_name in enumerate(self.clip_sensor_names):
            logger.info("Writing {} data".format(sensor_name))
            unsync[sensor_name] = list()
            token_list = sweeps_sd_tokens[sensor_name]
            for token in token_list:
                sd = self.nusc.get('sample_data', token)
                sensor_path = os.path.join(self.configs["dataroot"], sd['filename'])
                view_path = os.path.join(clip_root, sensor_name)
                # camera_xxx
                if "lidar" not in sensor_name:
                    shutil.copy(sensor_path, os.path.join(view_path, str(round(sd['timestamp'] / 1000)) + '.jpg'))
                # lidar_top
                else:
                    pointcloud = np.fromfile(sensor_path, dtype=np.double, count=-1).reshape([-1, 4])
                    pointcloud = np.hstack([pointcloud, np.zeros((pointcloud.shape[0], 2))])
                    pointcloud.tofile(os.path.join(view_path, str(round(sd['timestamp'] / 1000)) + '.bin'))
                unsync[self.clip_sensor_names[index]].append(round(sd['timestamp'] / 1000))
            logger.info("Done {}".format(sensor_name))
        return unsync

    def get_sensor_tokens(self, records):
        sweeps_sd_tokens = dict()

        # loop for each sensor
        for idx, sensor_name in enumerate(self.clip_sensor_names):
            sweeps_sd_tokens[sensor_name] = list()
            # first frame token
            sensor_sd_token = records[0]['data'][self.scene_sensor_names[idx]]
            sweeps_sd_tokens[self.clip_sensor_names[idx]].append(sensor_sd_token)
            sensor_sd = self.nusc.get('sample_data', sensor_sd_token)
            # if has next frame
            while sensor_sd['next']:
                sweeps_sd_tokens[self.clip_sensor_names[idx]].append(sensor_sd['next'])
                sensor_sd = self.nusc.get('sample_data', sensor_sd['next'])
        return sweeps_sd_tokens

    def get_sync_info(self, unsync_timestamps_dict, ref_sensor, max_diff):
        """sync the unsync timestamps with reference sensor

        Args:
            unsync_timestamps_dict (dict): each key is sensor name and its value is list of unsync timestamps in milliseconds
            ref_sensor (string): reference sensor name, must be in unsync_timestamps_dict
            max_diff (int): max_diff in milliseconds for synchronization

        Returns:
            dict: each key is sensor name and its value is list of sync timestamps in milliseconds
        """
        # assert ref_sensor in unsync_timestamps_dict
        sync_timestamps_dict = {}
        for sensor_name in unsync_timestamps_dict.keys():
            sync_timestamps_dict[sensor_name] = []
        for ref_timestamp in unsync_timestamps_dict[ref_sensor]:
            sync_timestamp = []
            for sensor in unsync_timestamps_dict.keys():
                if sensor == ref_sensor:
                    sync_timestamp.append(ref_timestamp)
                else:
                    # this may be slower than np.searchsorted theoretically
                    # but it handles boundary issues automatically and is more readable
                    # the length of unsync_timestamps is at the magnitude of 10^2 to 10^4
                    search_idx = np.abs(
                        np.asarray(unsync_timestamps_dict[sensor]) - ref_timestamp
                    ).argmin()
                    diff_time = abs(
                        unsync_timestamps_dict[sensor][search_idx] - ref_timestamp
                    )
                    if diff_time <= max_diff:
                        sync_timestamp.append(unsync_timestamps_dict[sensor][search_idx])
                    # else:
                    #     logger.warning(
                    #         f"skipped, diff time = {diff_time} is larger than max_diff = {max_diff}"
                    #     )
            if len(sync_timestamp) == len(unsync_timestamps_dict.keys()):
                for sensor_name, timestamp in zip(unsync_timestamps_dict, sync_timestamp):
                    sync_timestamps_dict[sensor_name].append(timestamp)
            # else:
            #     logger.info("skip frame {}".format(ref_timestamp))
        return sync_timestamps_dict

    def get_nusc_map(self, scene):
        scene_name = scene["name"]
        wigo_path = os.path.join(self.configs["converted_dataroot"], scene_name, 'odometry/wigo.txt')
        wigo_np = np.loadtxt(wigo_path)
        mid_idx = int(wigo_np.shape[0] / 2) + 1
        mid_wigo = wigo_np[mid_idx]

        wigo_max = np.max(wigo_np, axis=0)
        wigo_min = np.min(wigo_np, axis=0)
        diff = wigo_max - wigo_min
        patch_center = (wigo_min[1] + diff[1]/2, wigo_min[2] + diff[2]/2)

        patch_w = diff[1] + 25
        patch_h = diff[2] + 25
        patch_size = (patch_h, patch_w)

        location = self.nusc.get('log', scene['log_token'])['location']
        ego_pose_t = mid_wigo[1:4].tolist()
        ego_pose_r = [mid_wigo[7]] + mid_wigo[4:7].tolist()

        vector_map = VectorizedLocalMap(self.configs["dataroot"], patch_size=patch_size)
        anns_results = vector_map.gen_vectorized_samples(location, ego_pose_t, ego_pose_r, patch_size, patch_center)

        gt_labels = np.array(anns_results['gt_vecs_label'])
        gt_vecs = anns_results['gt_vecs_pts_loc']

        gt_vec_list = []
        for i, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
            anno = dict(
                attrs=dict(
                    type=self.configs["map_classes"][gt_label]
                ),
                data=np.array(list(gt_vec.coords)).tolist(),
                id=-1,
                luid='auto',
                point_attrs=[[] for i in range(len(list(gt_vec.coords)))],
                shape_type='polyline',
                struct_type='parsing',
                track_id=-1
            )
            gt_vec_list.append(anno)
        return gt_vec_list

    def convert(self, scene_name):
        scene = self.get_scene_by_name(scene_name)

        # prepare output dir
        clip_root = os.path.join(self.configs["converted_dataroot"], scene_name)
        if not os.path.exists(clip_root):
            os.makedirs(clip_root)

        # get timestamp start end
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']

        # nuscenes timestamp，μs to ms
        start_time = round(self.nusc.get('sample', first_sample_token)['timestamp'] / 1000)
        end_time = round(self.nusc.get('sample', last_sample_token)['timestamp'] / 1000)

        # init clip attribute dict
        attr_dict = {
            "start_time": start_time,
            "end_time": end_time,
            "status": "init",
            "calibration": dict(),
        }

        # prepare records and sort by timestamp
        records = [samp for samp in self.samples if
                   self.nusc.get("scene", samp["scene_token"])["name"] in scene_name]
        records.sort(key=lambda x: (x['timestamp']))

        sweeps_sd_tokens = self.get_sensor_tokens(records)

        self.write_odometry(clip_root, sweeps_sd_tokens)

        unsync = self.write_sensors(sweeps_sd_tokens, clip_root)
        attr_dict.update({"unsync": unsync})

        sync = self.get_sync_info(unsync, "camera_front", 40)
        attr_dict.update({"sync": sync})

        calibration = self.get_calibration(records)
        attr_dict.update({"calibration": calibration})

        with open(os.path.join(clip_root, 'attribute.json'), 'w') as jf:
            json.dump(attr_dict, jf, indent=4, ensure_ascii=False)

        nuscenes_map = self.get_nusc_map(scene)
        map_dir = os.path.join(clip_root, self.configs["cama_configs"]["result_dir"])
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        nuscenes_map_path = os.path.join(map_dir, "map_nuscenes.json")
        with open(nuscenes_map_path, 'w') as f:
            json.dump(nuscenes_map, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':

    configs = dict()
    configs["version"] = 'v1.0-test'
    configs["dataroot"] = 'xxx/nuScenes/origin/'
    configs["converted_dataroot"] = 'xxx/converted_nuscenes'
    configs["map_classes"] = ['lane_marking', 'Road_teeth', 'Crosswalk_Line']
    configs["cama_configs"] = dict()
    configs["cama_configs"]["result_dir"] = "maps"

    s2c = nuScenes2Clip(configs)

    scene_name = "scene-0550"
    s2c.convert(scene_name)
