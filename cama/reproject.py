import cv2
import numpy as np
from os.path import join
from horizon_driving_dataset import DatasetReader


class BaseManager:
    def __init__(self):
        pass

    @staticmethod
    def get_color_maps():
        color_maps = {"Road_teeth": np.array([235, 73, 127]),
                      "lane_marking": np.array([211, 211, 211]),
                      "Stop_Line": np.array([211, 211, 211]),
                      "Crosswalk_Line": np.array([255, 215, 0])}
        return color_maps


class MapManager(BaseManager):
    def __init__(self,):
        super(MapManager, self).__init__()
        self.solution = 0.1  # meter per pixel
        self.center_x = 0  # meter
        self.center_y = 0  # meter
        self.map_width = 300  # meter
        self.map_height = 300  # meter
        self.crop_dict = {}
        self.crop_dict["x_min"] = -50
        self.crop_dict["x_max"] = 50
        self.crop_dict["y_min"] = -100
        self.crop_dict["y_max"] = 100
        self.crop_dict["z_min"] = -200
        self.crop_dict["z_max"] = 200

    def pixel2world_xy(self, pixel_xy):
        worlds_xy = np.zeros_like(pixel_xy)
        worlds_xy[:, 0] = pixel_xy[:, 1] * self.solution - self.map_width / 2 + self.center_x
        worlds_xy[:, 1] = pixel_xy[:, 0] * self.solution - self.map_height / 2 + self.center_y
        return worlds_xy

    def load_3d_instance_maps(self, maps_2d):
        instance_list = []
        for instance in maps_2d:
            instance_class = instance["attrs"]["type"]
            line_points = instance['data']
            if len(line_points) <= 1:  # too short, neglect
                continue
            line_points = np.array(line_points).astype(np.float32)

            # interpolate line_points
            inter_line_points = []
            length = np.linalg.norm(line_points[1:] - line_points[:-1], axis=-1)
            for i in range(len(length)):
                start_point = line_points[i]
                end_point = line_points[i+1]
                num = int(length[i] / self.solution)
                if num == 0:
                    continue
                else:
                    for j in range(num):
                        inter_line_points.append(start_point + (end_point - start_point) / num * j)
            inter_line_points = np.array(inter_line_points)

            line_points_h = np.zeros_like(inter_line_points[:, 0])
            # add height
            world_xy = inter_line_points
            world_xyz = np.concatenate((world_xy, line_points_h[:, None]), axis=-1).reshape(-1, 3)
            instance_list.append({"class": instance_class, "points": world_xyz})
        return instance_list

    def calculate_3d_instance_maps(self, bev_height, maps_2d):
        instance_list = []
        for instance in maps_2d:
            instance_class = instance["attrs"]["type"]
            line_points = instance['data']
            if len(line_points) <= 1:  # too short, neglect
                continue
            line_points = np.array(line_points).astype(np.float32)

            # interpolate line_points
            inter_line_points = []
            length = np.linalg.norm(line_points[1:] - line_points[:-1], axis=-1)
            for i in range(len(length)):
                start_point = line_points[i]
                end_point = line_points[i+1]
                num = int(length[i] / self.solution)
                if num == 0:
                    continue
                else:
                    for j in range(num):
                        inter_line_points.append(start_point + (end_point - start_point) / num * j)
            inter_line_points = np.array(inter_line_points)

            # get bev height
            line_points_pixel = inter_line_points.round().astype(np.uint16)
            line_points_pixel = line_points_pixel[:, ::-1]  # H * W
            line_points_pixel = line_points_pixel.clip(0, bev_height.shape[0]-1)
            line_points_h = bev_height[line_points_pixel[:, 0], line_points_pixel[:, 1]]

            # add height
            world_xy = self.pixel2world_xy(inter_line_points)
            world_xyz = np.concatenate((world_xy, line_points_h[:, None]), axis=-1).reshape(-1, 3)
            instance_list.append({"class": instance_class, "points": world_xyz})

        return instance_list

    def transform_3d_instance_maps(self, maps, transform):
        instance_list = []
        for instance in maps:
            instance_class = instance["class"]
            points = instance["points"]
            points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=-1)
            points = (transform @ points.T).T
            instance_list.append({"class": instance_class, "points": points[:, :3]})
        return instance_list

    def crop_3d_instance_maps(self, maps, crop_dict=None):
        crop_dict = crop_dict if crop_dict is not None else self.crop_dict
        instance_list = []
        for instance in maps:
            instance_class = instance["class"]
            points = instance["points"]
            mask = (points[:, 0] >= crop_dict["x_min"]) & (points[:, 0] <= crop_dict["x_max"]) & \
                (points[:, 1] >= crop_dict["y_min"]) & (points[:, 1] <= crop_dict["y_max"]) & \
                (points[:, 2] >= crop_dict["z_min"]) & (points[:, 2] <= crop_dict["z_max"])
            points = points[mask]
            num_points = points.shape[0]
            if num_points > 0:
                instance_list.append({"class": instance_class, "points": points})
        return instance_list

    def save_pcd(self, maps, pcd_path):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        # save all instance points to single pcd with colors
        points_list = []
        colors_list = []
        for instance in maps:
            points = instance["points"]
            colors = self.get_color_maps()[instance["class"]]
            # extend colors to points shape
            colors = np.tile(colors, (points.shape[0], 1))
            points_list.append(points)
            colors_list.append(colors)

        points = np.concatenate(points_list, axis=0)
        colors = np.concatenate(colors_list, axis=0)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.)
        o3d.io.write_point_cloud(pcd_path, pcd)

    def save_xyz(self, maps, xyz_path):
        # save all points with only xyz
        points_list = []
        for instance in maps:
            points = instance["points"]
            points_list.append(points)
        points = np.concatenate(points_list, axis=0)
        np.savetxt(xyz_path, points, fmt="%.3f")


class CameraManager(BaseManager):
    def __init__(self, clip_path, camera_name, output_size=(540, 960), undisort=True):
        super(CameraManager, self).__init__()
        dr = DatasetReader(clip_path)
        self.dr = dr
        self.clip_path = clip_path
        self.camera_name = camera_name
        self.chassis2camera = dr.get_extrinsic("chassis", camera_name)
        intrinsics = dr.get_intrinsics(camera_name)
        self.K_origin = intrinsics["K"]
        self.d_origin = intrinsics["d"]
        self.width_origin = intrinsics["width"]
        self.height_origin = intrinsics["height"]
        self.width = output_size[1]
        self.height = output_size[0]
        if undisort:
            self.d = []
        self.K = self.K_origin.copy()
        self.K[0, :] = self.K[0, :] * self.width / self.width_origin
        self.K[1, :] = self.K[1, :] * self.height / self.height_origin

    def get_chassis2camera(self):
        return self.chassis2camera

    def project_to_image(self, maps):
        vu_list = []
        for instance in maps:
            points = instance["points"]
            points = (self.K @ points.T).T
            mask_z = points[:, 2] > 0
            points = points[:, :] / points[:, 2:]
            # mask for z>0 and u,v within image
            mask = (points[:, 2] > 0) & \
                   (points[:, 0] >= 0) & (points[:, 0] < self.width) & \
                   (points[:, 1] >= 0) & (points[:, 1] < self.height)
            mask = mask & mask_z
            points = points[mask]
            num_points = points.shape[0]
            if num_points > 0:
                uv_points = points[:, :2]
                vu_points = uv_points[:, ::-1]
                vu_list.append({"class": instance["class"], "points": vu_points})
        return vu_list

    def index2timestamp(self, index, sync):
        sync_str = "sync" if sync else "unsync"
        timestamp_int = self.dr.attribute[sync_str][self.camera_name][index]
        return timestamp_int

    def get_image_path(self, index, sync):
        timestamp_int = self.index2timestamp(index, sync)
        image_path = join(self.clip_path, self.camera_name, f"{timestamp_int}.jpg")
        return image_path

    def get_instance_path(self, index, sync=True):
        timestamp_int = self.index2timestamp(index, sync)
        instance_path = join(self.clip_path, f"lane_ins_{self.camera_name}", f"{timestamp_int}.png")
        return instance_path

    def read_resized_instance_by_index(self, index, sync=True):
        instace_path = self.get_instance_path(index, sync=sync)
        origin_instance = cv2.imread(instace_path, cv2.IMREAD_ANYDEPTH)
        instance_image = self.resize_image(origin_instance, interpolation=cv2.INTER_NEAREST)
        return instance_image

    def read_resized_image_by_index(self, index, sync=True):
        image_path = self.get_image_path(index, sync)
        return self.read_resized_image(image_path)

    def resize_image(self, image, interpolation=cv2.INTER_LINEAR):
        new_size = (self.width, self.height)
        if self.d == []:
            distortion = self.d_origin
        else:
            distortion = self.d
        mapx, mapy = cv2.initUndistortRectifyMap(self.K_origin, distortion, None, self.K, new_size, cv2.CV_32FC1)
        undistorted_resized_image = cv2.remap(image, mapx, mapy, interpolation=interpolation)
        return undistorted_resized_image

    def read_resized_image(self, image_path):
        image = cv2.imread(image_path)
        return self.resize_image(image)

    def render_maps(self, image, maps_2d):
        for instance in maps_2d:
            points = instance["points"]
            points = points.astype(np.int32)
            class_name = instance["class"]
            if class_name != "lane_marking":
                class_name = "Crosswalk_Line"
            color = self.get_color_maps()[class_name]
            color_tuple = tuple(color[::-1].tolist())
            for point in points:
                cv2.circle(image, (point[1], point[0]), 2, color_tuple, -1)
        return image
