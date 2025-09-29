import numpy as np
from skimage.morphology import skeletonize
from shapely.geometry import LineString
from shapely.geometry import Point
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment


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

class MetricEvaluation(BaseManager):
    def __init__(self):
        super(MetricEvaluation, self).__init__()
        self.max_match_pairs = 8

    def load_instance(self, instances_png):
        encoded_instances = np.unique(instances_png)
        # remove background
        encoded_instances = encoded_instances[encoded_instances != 0]
        instances = {}
        for encoded_instance in encoded_instances:
            semantic_id = encoded_instance // 1000
            instance_id = encoded_instance % 1000
            mask = instances_png == encoded_instance
            if semantic_id not in instances:
                instances[semantic_id] = []
            instances[semantic_id].append(mask)
        return instances

    def semantic_id_remap(self, semantic_id):
        if semantic_id in [0, 1, 2, 3, 4, 5, 6]:
            return "lane_marking"
        elif semantic_id == 7:
            return "Road_teeth"
        else:
            return "Unknown"

    def load_instance_vectors(self, instances_png):
        instances = self.load_instance(instances_png)
        vectors = self.instance_2_vectors(instances)
        return vectors

    def match_skeleton(self, skeleton_image, skeleton_projected):
        # match skeleton with hungarian algorithm
        # Step 1: Matrix Formation
        num_points_image = len(skeleton_image)
        num_points_projected = len(skeleton_projected)

        # Initialize the cost matrix with high values
        cost_matrix = np.full((num_points_image, num_points_projected), 500.0)

        for i in range(num_points_image):
            for j in range(num_points_projected):
                distance = self.compute_curve_distance_only(skeleton_image[i], skeleton_projected[j])
                if distance is not None:
                    cost_matrix[i][j] = distance

        # Step 2: Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Step 3: return the matched pairs
        matched_pairs = []
        for i in range(len(row_ind)):
            if cost_matrix[row_ind[i]][col_ind[i]] < 300.0:
                matched_pairs.append((row_ind[i], col_ind[i]))
        return matched_pairs

    def masks2skeleton(self, instance_image):
        vectors = []
        for semantic_id, masks in instance_image.items():
            semantic_name = self.semantic_id_remap(semantic_id)
            for mask in masks:
                vector = {}
                vector["class"] = semantic_name
                skeleton = skeletonize(mask)
                vector["points"] = np.argwhere(skeleton)
                vectors.append(vector)
        return vectors

    def cut_linestring(self, linestring_image, linestring_projected):
        # Check if the linestrings have enough coordinates
        if len(linestring_projected.coords) < 2:
            raise ValueError("The linestring_projected should have at least two coordinates.")

        # 1. Calculate the normals at the start and end of the linestring_projected
        start, end = linestring_projected.coords[0], linestring_projected.coords[-1]

        # Calculate vectors for start and end points
        vector_start = (linestring_projected.coords[1][0] - start[0], linestring_projected.coords[1][1] - start[1])
        vector_end = (linestring_projected.coords[-2][0] - end[0], linestring_projected.coords[-2][1] - end[1])

        # Calculate normals by swapping x and y coordinates and inverting one of them
        # Extending the normals in both directions by a large number (e.g., 1e5) to ensure intersection
        normal_start = LineString([(start[0] - 1e5*vector_start[1], start[1] + 1e5*vector_start[0]),
                                   (start[0] + 1e5*vector_start[1], start[1] - 1e5*vector_start[0])])
        normal_end = LineString([(end[0] - 1e5*vector_end[1], end[1] + 1e5*vector_end[0]),
                                (end[0] + 1e5*vector_end[1], end[1] - 1e5*vector_end[0])])

        # 2. Find the intersections of these normals with the linestring_image
        intersection_start = linestring_image.intersection(normal_start)
        intersection_end = linestring_image.intersection(normal_end)

        # Check if intersections exist
        if intersection_start.is_empty or intersection_end.is_empty:
            raise ValueError("No intersections found with the normals.")

        # 3. Cut the linestring_image at these intersection points
        cut_point_start = intersection_start.coords[0]
        cut_point_end = intersection_end.coords[0]

        # Split the linestring_image using the intersection points
        cut_linestring_image = linestring_image.split(cut_point_start)[0].split(cut_point_end)[-1]

        # Assuming the cut_linestring_projected is the segment between the two intersection points
        cut_linestring_projected = LineString([cut_point_start, cut_point_end])

        return cut_linestring_image, cut_linestring_projected

    def compute_normals(self, spline_tck, search_length=20.0):
        u_new = np.linspace(0.2, 0.8, 60)
        # Compute the tangent vectors (derivatives) of the spline
        tangent_x, tangent_y = splev(u_new, spline_tck, der=1)

        # Compute the normal vectors
        normals_x = -tangent_y
        normals_y = tangent_x

        # Normalize the normals
        norm_lengths = np.sqrt(normals_x**2 + normals_y**2)
        normals_x /= norm_lengths
        normals_y /= norm_lengths
        # for each normal, compute LineString with length -5, 5
        linestring_list = []
        for idx, u in enumerate(u_new):
            x, y = splev(u, spline_tck)
            nv = np.array([normals_x[idx], normals_y[idx]])
            left_point = np.array([x - search_length * nv[0], y - search_length * nv[1]])
            right_point = np.array([x + search_length * nv[0], y + search_length * nv[1]])
            linestring = LineString([left_point, right_point])
            linestring_list.append(linestring)
        return linestring_list

    def normals2line_distance(self, normals, linestring):
        distance = []
        for normal in normals:
            intersection = normal.intersection(linestring)
            if intersection.is_empty:
                continue
            elif not isinstance(intersection, Point):
                continue
            else:
                start_point_a = normal.centroid
                end_point_b = intersection
                error_linestring = LineString([start_point_a, end_point_b])
                distance.append(error_linestring)
        return distance

    def vis_spline(self, vectors, spline_tck, image_shape=(540, 960, 3)):
        # helper function to visualize the spline
        canvas = np.zeros(image_shape, dtype=np.uint8)
        # canvas[vectors[:, 0], vectors[:, 1], :] = (255, 255, 255)
        u_new = np.linspace(0, 1, 100)
        x, y = splev(u_new, spline_tck)
        points = np.stack([x, y], axis=-1).astype(np.int32)
        for point in points:
            canvas[point[0], point[1], :] = (255, 255, 255)
        return canvas

    def fit_spline(self, vector):
        length = vector.shape[0]
        spline_tck, u = splprep([vector[:, 0], vector[:, 1]], s=float(length)*10)
        return spline_tck, u

    def compute_curve_distance(self, vector_image, vector_projected):
        spline_tck_image, _ = self.fit_spline(vector_image)
        # vis_spline = self.vis_spline(vector_projected, spline_tck_projected)
        # cv2.imwrite("tmp/vis_spline.jpg", vis_spline)
        linestring_projected = LineString(vector_projected)
        normals_image = self.compute_normals(spline_tck_image)
        distance_list = self.normals2line_distance(normals_image, linestring_projected)
        return distance_list

    def distance2error(self, distance_list):
        if len(distance_list) == 0:
            return None
        length = 0
        number = 0
        for distance in distance_list:
            length += distance.length
            number += 1
        mean_distance = length / number
        return mean_distance

    def compute_curve_distance_only(self, curve_image, curve_projected):
        distance_list = self.compute_curve_distance(curve_image, curve_projected)
        mean_distance = self.distance2error(distance_list)
        return mean_distance

    def filter_line_increasing(self, line):
        # make sure the line is strictly increasing in x
        line = line[line[:, 1].argsort()]
        _, idx = np.unique(line[:, 1], return_index=True)
        line = line[idx]
        return line

    def pick_a_class(self, vectors_dict, class_name):
        vectors = []
        for vector in vectors_dict:
            if vector["class"] == class_name:
                vectors.append(vector["points"])
        return vectors

    def vectors_float2int(self, vectors):
        for vector in vectors:
            vector["points"] = vector["points"].astype(np.int32)
            vector["points"] = np.unique(vector["points"], axis=0)
        return vectors

    def remove_too_short_vector(self, vector_list, length=10):
        new_vector_list = []
        for vector in vector_list:
            if len(vector) >= length:
                new_vector_list.append(vector)
        return new_vector_list

    def calculate_sre(self, vectors_image, vectors_projected):
        errors = []
        sres_vis = []
        recalls = []
        precisions = []
        # filter by class
        class_names = list(self.get_color_maps().keys())[1:2]
        for class_name in class_names:
            vectors_image_list = self.pick_a_class(vectors_image, class_name)
            vectors_projected_list = self.pick_a_class(vectors_projected, class_name)

            # filter length
            vectors_image_list = self.remove_too_short_vector(vectors_image_list)
            vectors_projected_list = self.remove_too_short_vector(vectors_projected_list)

            # skip this class if no instance
            if len(vectors_image_list) == 0 or len(vectors_projected_list) == 0:
                continue

            matched_pairs = self.match_skeleton(vectors_image_list, vectors_projected_list)
            # matched_pairs = matched_pairs[:self.max_match_pairs]

            tp = len(matched_pairs) # pred 和 gt 同时存在并且匹配的数量
            fn = len(vectors_image_list) - tp # gt 存在但是 pred 中未匹配的数量
            fp = len(vectors_projected_list) - tp # pred 中存在但是 gt 中未匹配的数量
            precision = float(tp) / float(tp+fp)
            precisions.append(precision)
            recall = float(tp) / float(tp+fn)
            recalls.append(recall)


            for image_id, projected_id in matched_pairs:
                vector_image = vectors_image_list[image_id]
                vector_projected = vectors_projected_list[projected_id]
                distance_list = self.compute_curve_distance(vector_image, vector_projected)
                error = self.distance2error(distance_list)
                if error is not None:
                    errors.append(error)
                    sres_vis.append(distance_list)
        if len(errors) == 0:
            return None, None, None, None
        precisions = np.array(precisions).mean()
        recalls = np.array(recalls).mean()
        return sres_vis, errors, precisions, recalls
