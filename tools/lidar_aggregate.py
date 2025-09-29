#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuse nuScenes LiDAR with CAMAv2 camera poses (with height) for accurate colored 3D aggregation.

Usage:
python lidar_aggregate.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-test \
  --scene-name scene-0550 \
  --sfm-dir /path/to/cama_label \
"""

import os
import argparse
import numpy as np
import cv2
from bisect import bisect_left
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from typing import Union
from tqdm import tqdm



DEFAULT_CAMS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

CONVERTED_CAMS = [
    "camera_front",
    "camera_front_right",
    "camera_rear_right",
    "camera_rear",
    "camera_rear_left",
    "camera_front_left",
]

# ------------------------------
# Basic scene iteration helpers
# ------------------------------
def get_scene_by_name(nusc, scene_name):
    for s in nusc.scene:
        if s["name"] == scene_name:
            return s
    raise ValueError(f"Scene name '{scene_name}' not found.")


def collect_scene_sample_tokens(nusc, scene_rec):
    tokens = []
    token = scene_rec["first_sample_token"]
    while token:
        tokens.append(token)
        sample = nusc.get("sample", token)
        token = sample["next"]
    return tokens, set(tokens)


def list_lidar_sweeps_in_scene(nusc, scene_rec, sensor_channel="LIDAR_TOP"):
    """Return sorted list of (timestamp_us, sd_token) for all LiDAR sweeps in this scene."""
    _, sample_tokens_set = collect_scene_sample_tokens(nusc, scene_rec)
    first_sample = nusc.get("sample", scene_rec["first_sample_token"])
    if sensor_channel not in first_sample["data"]:
        return []

    # move to head
    sd = nusc.get("sample_data", first_sample["data"][sensor_channel])
    while sd["prev"] != "":
        prev_sd = nusc.get("sample_data", sd["prev"])
        if prev_sd["sample_token"] not in sample_tokens_set:
            break
        sd = prev_sd

    out = []
    while True:
        if sd["sample_token"] in sample_tokens_set:
            out.append((sd["timestamp"], sd["token"]))
        if sd["next"] == "":
            break
        nxt = nusc.get("sample_data", sd["next"])
        if nxt["sample_token"] not in sample_tokens_set:
            break
        sd = nxt

    out.sort(key=lambda t: t[0])
    return out


def list_cam_sd_in_scene(nusc, scene_rec, camera_channel):
    """Return sorted list of (timestamp_us, sd_token) for all camera sds in this scene."""
    _, sample_tokens_set = collect_scene_sample_tokens(nusc, scene_rec)

    # Start from the first keyframe's camera sd
    first_sample = nusc.get("sample", scene_rec["first_sample_token"])
    if camera_channel not in first_sample["data"]:
        return []

    sd = nusc.get("sample_data", first_sample["data"][camera_channel])

    # Move to head
    while sd["prev"] != "":
        prev_sd = nusc.get("sample_data", sd["prev"])
        if prev_sd["sample_token"] not in sample_tokens_set:
            break
        sd = prev_sd

    sds = []
    while True:
        if sd["sample_token"] in sample_tokens_set:
            sds.append((sd["timestamp"], sd["token"]))
        if sd["next"] == "":
            break
        nxt = nusc.get("sample_data", sd["next"])
        if nxt["sample_token"] not in sample_tokens_set:
            break
        sd = nxt

    sds.sort(key=lambda t: t[0])
    return sds


# ------------------------------
# Timestamp normalization & NN
# ------------------------------
def to_microseconds(ts_val):
    """
    Normalize timestamps to microseconds.
    Accepts float/int seconds (e.g., 1532402925.718276) or int microseconds (e.g., 1532402925718276).
    """
    if isinstance(ts_val, str):
        ts_val = float(ts_val.strip())
    if isinstance(ts_val, float):
        # likely seconds
        return int(round(ts_val * 1e6))
    if isinstance(ts_val, int):
        # if it's too small, treat as seconds
        if ts_val < 10_000_000_000:  # 1e10 ~ 2001 in microseconds; nuScenes ~1e15
            return ts_val * int(1e6)
        return ts_val
    raise ValueError(f"Unsupported timestamp type: {type(ts_val)}")


def nearest_item(sorted_pairs, ts_us):
    """
    Binary search nearest (timestamp, token) in a list sorted by timestamp.
    Returns (timestamp, token, abs_diff_us). If list empty, returns (None, None, None).
    """
    if not sorted_pairs:
        return None, None, None
    ts_list = [t for t, _ in sorted_pairs]
    i = bisect_left(ts_list, ts_us)
    candidates = []
    if i < len(sorted_pairs):
        candidates.append(sorted_pairs[i])
    if i > 0:
        candidates.append(sorted_pairs[i - 1])
    # pick min abs diff
    best = min(candidates, key=lambda p: abs(p[0] - ts_us))
    return best[0], best[1], abs(best[0] - ts_us)


# ------------------------------
# Projection (nuScenes-only)
# ------------------------------
def project_lidar_to_cam_sparse(
    nusc, lidar_sd, cam_sd, max_range=None, z_min=0.1, z_max=150.0
):
    """
    Project a LiDAR sweep into a camera using ONLY nuScenes poses/calib.
    Returns sparse per-pixel depth samples as arrays (u, v, z),
    where z is in meters in the camera coords at cam_sd time (after z-buffer).
    """
    # LiDAR -> world at LiDAR time
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    T_lidar_to_ego = transform_matrix(
        lidar_cs["translation"], Quaternion(lidar_cs["rotation"]), inverse=False
    )
    T_ego_to_world_lidar = transform_matrix(
        lidar_pose["translation"], Quaternion(lidar_pose["rotation"]), inverse=False
    )

    lidar_path = os.path.join(nusc.dataroot, lidar_sd["filename"])
    pc = LidarPointCloud.from_file(lidar_path)
    # Transform to world
    pc.transform(T_lidar_to_ego)
    pc.transform(T_ego_to_world_lidar)
    pts_world = pc.points[:3, :].T.astype(np.float32)

    # Optional radial crop in world coords (about origin)
    if max_range and max_range > 0:
        r2 = np.einsum("ij,ij->i", pts_world, pts_world)
        keep = r2 <= (float(max_range) ** 2)
        pts_world = pts_world[keep]
        if pts_world.shape[0] == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32),
            )

    # world -> ego(cam time) -> cam
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    T_ego_to_world_cam = transform_matrix(
        cam_pose["translation"], Quaternion(cam_pose["rotation"]), inverse=False
    )
    T_world_to_ego_cam = np.linalg.inv(T_ego_to_world_cam)
    T_sensor_to_ego_cam = transform_matrix(
        cam_cs["translation"], Quaternion(cam_cs["rotation"]), inverse=False
    )
    T_ego_to_cam = np.linalg.inv(T_sensor_to_ego_cam)

    N = pts_world.shape[0]
    pts_w_h = np.concatenate(
        [pts_world, np.ones((N, 1), dtype=np.float32)], axis=1
    ).T  # (4,N)
    pts_cam = T_ego_to_cam @ (T_world_to_ego_cam @ pts_w_h)
    xyz_cam = pts_cam[:3, :].T

    # Depth filter
    z = xyz_cam[:, 2]
    valid = (z > z_min) & (z < z_max)
    if not np.any(valid):
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
        )

    # Project with nuScenes K at native size
    K = np.array(cam_cs["camera_intrinsic"], dtype=np.float32)
    xv = xyz_cam[valid, 0] / z[valid]
    yv = xyz_cam[valid, 1] / z[valid]
    uv = (K @ np.stack([xv, yv, np.ones_like(xv)], axis=0)).T
    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)

    # Image size
    img = cv2.imread(os.path.join(nusc.dataroot, cam_sd["filename"]), cv2.IMREAD_COLOR)
    if img is None:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
        )
    H, W = img.shape[0], img.shape[1]
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_img):
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
        )

    u = u[in_img]
    v = v[in_img]
    z_sel = z[valid][in_img]

    # per-pixel z-buffer (keep nearest)
    pix_id = v * W + u
    order = np.lexsort((z_sel, pix_id))  # primary by pix_id, secondary by z
    pix_id_sorted = pix_id[order]
    u_sorted = u[order]
    v_sorted = v[order]
    z_sorted = z_sel[order]
    first = np.ones_like(pix_id_sorted, dtype=bool)
    first[1:] = pix_id_sorted[1:] != pix_id_sorted[:-1]

    return u_sorted[first], v_sorted[first], z_sorted[first]

# ------------------------------
# SfM pose loader (TUM lines)
# ------------------------------
def load_sfm_tum(path_txt):
    """
    Load TUM pose list: lines of `timestamp tx ty tz qx qy qz qw`.
    Returns list of dicts with keys: ts_us, R(3x3), t(3,), and 4x4 T_cam_to_world.
    """
    poses = []
    if not os.path.exists(path_txt):
        return poses
    with open(path_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            ts, tx, ty, tz, qx, qy, qz, qw = parts
            ts_us = to_microseconds(ts)
            q = Quaternion(
                float(qw), float(qx), float(qy), float(qz)
            )  # pyquaternion expects w,x,y,z
            R = q.rotation_matrix.astype(np.float32)
            t = np.array([float(tx), float(ty), float(tz)], dtype=np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = t
            poses.append(dict(ts_us=ts_us, R=R, t=t, T=T))
    poses.sort(key=lambda d: d["ts_us"])
    return poses


# ------------------------------
# Back-projection with SfM pose
# ------------------------------
def backproject_uvz_to_world_sfm(u, v, z, K, T_cam_to_world_sfm, rgb_image):
    """
    Given sparse pixels (u,v) with metric depths z (in camera coords),
    back-project to SfM world: X_cam = z * K^{-1} [u v 1]^T ; X_w = R*X_cam + t
    Returns (xyz_world (M,3), rgb (M,3 uint8))
    """
    if u.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    H, W = rgb_image.shape[0], rgb_image.shape[1]
    # Safety: clip to image bounds
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    Kinv = np.linalg.inv(K)
    ones = np.ones_like(z)
    pix = np.stack(
        [u.astype(np.float32), v.astype(np.float32), ones.astype(np.float32)], axis=0
    )  # (3,M)
    rays = Kinv @ pix  # (3,M)
    X_cam = (rays * z[np.newaxis, :]).T  # (M,3)

    R = T_cam_to_world_sfm[:3, :3]
    t = T_cam_to_world_sfm[:3, 3]
    X_w = (R @ X_cam.T).T + t[np.newaxis, :]

    # Fetch RGB (convert BGR->RGB)
    img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    colors = img_rgb[v, u]  # (M,3)
    return X_w.astype(np.float32), colors.astype(np.uint8)


def write_ascii_ply_xyz_rgb(path, xyz, rgb):
    xyz = np.asarray(xyz)
    rgb = np.asarray(rgb).astype(np.uint8)
    N = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(
        "Fuse LiDAR with CAMAv2 SfM poses for accurate height-aware colored aggregation."
    )
    ap.add_argument(
        "--dataroot",
        required=True,
        type=str,
        help="nuScenes dataroot",
    )
    ap.add_argument(
        "--version", default="v1.0-trainval", type=str, help="nuScenes version"
    )
    ap.add_argument("--scene-name", required=True, type=str, help="Scene name, e.g. scene-0061")
    ap.add_argument("--sensor", default="LIDAR_TOP", type=str)
    ap.add_argument(
        "--cams", nargs="*", default=DEFAULT_CAMS, help="Camera channels to use."
    )
    ap.add_argument(
        "--sfm-dir",
        required=True,
        type=str,
        help="Directory containing scmv_{CAM}.txt files.",
    )
    ap.add_argument(
        "--min-depth",
        type=float,
        default=0.2,
        help="Min valid depth (m) for LiDAR->cam samples.",
    )
    ap.add_argument(
        "--max-depth",
        type=float,
        default=150.0,
        help="Max valid depth (m) for LiDAR->cam samples.",
    )
    ap.add_argument(
        "--max-range",
        type=float,
        default=None,
        help="Optional radial crop (m) before projection.",
    )
    ap.add_argument(
        "--dt-lidar-ms",
        type=float,
        default=50.0,
        help="Max |Δt| between SfM cam ts and chosen LiDAR sweep.",
    )
    ap.add_argument(
        "--dt-cam-ms",
        type=float,
        default=50.0,
        help="Max |Δt| between SfM cam ts and chosen nuScenes camera SD (for K/image).",
    )
    ap.add_argument(
        "--every", type=int, default=1, help="Use every k-th SfM frame per camera."
    )
    ap.add_argument("--out", type=str, help="Output .ply or .npy")
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Scene selection
    scene_rec = get_scene_by_name(nusc, args.scene_name)

    # Check if --out argument is provided
    if not args.out:
        # If not provided, set default filename based on scene_name
        scene_name = args.scene_name or scene_rec["name"]
        args.out = os.path.join(
            "output",
            f"{scene_name}_lidar_aggregated.ply",  # You can adjust the format based on your needs
        )

    # Pre-index LiDAR sweeps & camera sds
    lidar_sweeps = list_lidar_sweeps_in_scene(
        nusc, scene_rec, args.sensor
    )  # [(ts, token)...]

    cam_sds = {}
    for cam in args.cams:
        cam_sds[cam] = list_cam_sd_in_scene(nusc, scene_rec, cam)
        if not cam_sds[cam]:
            print(f"[Warn] No camera sds found for {cam} in scene, skipping this cam.")

    # Load SfM TUM pose files per camera
    sfm_files = {
        cam: os.path.join(args.sfm_dir, scene_name, "odometry", f"scmv_{cam}.txt")
        for cam in CONVERTED_CAMS
    }
    sfm_poses = {cam: load_sfm_tum(path) for cam, path in sfm_files.items()}
    total_sfm = sum(len(v) for v in sfm_poses.values())
    if total_sfm == 0:
        raise RuntimeError(
            "No SfM poses loaded. Check --sfm-dir and file naming: scmv_{CAM}.txt"
        )

    # keep_set = set(args.keep_ids)
    dt_lidar_us = (
        int(round(max(args.dt_lidar_ms, 0) * 1000.0)) if args.dt_lidar_ms > 0 else None
    )
    dt_cam_us = (
        int(round(max(args.dt_cam_ms, 0) * 1000.0)) if args.dt_cam_ms > 0 else None
    )

    # Accumulators (SfM-world)
    all_xyz = []
    all_rgb = []

    # Process per camera
    for cam, converted_cam in zip(args.cams, CONVERTED_CAMS):
        poses = sfm_poses.get(converted_cam, [])
        if not poses:
            continue
        sd_list = cam_sds.get(cam, [])
        if not sd_list:
            continue

        print(f"[{converted_cam}] SfM frames: {len(poses)}")
        for idx, pose in enumerate(tqdm(poses, desc=f"SfM frames for {converted_cam}")):
            if idx % max(1, int(args.every)) != 0:
                continue

            ts_us = pose["ts_us"]

            # nearest LiDAR sweep
            t_lidar, lidar_sd_token, diff_lidar = nearest_item(lidar_sweeps, ts_us)
            if lidar_sd_token is None:
                continue
            if (dt_lidar_us is not None) and (diff_lidar > dt_lidar_us):
                continue
            lidar_sd = nusc.get("sample_data", lidar_sd_token)

            # nearest camera sd (for K and image)
            t_cam, cam_sd_token, diff_cam = nearest_item(sd_list, ts_us)
            if cam_sd_token is None:
                continue
            if (dt_cam_us is not None) and (diff_cam > dt_cam_us):
                continue
            cam_sd = nusc.get("sample_data", cam_sd_token)

            # 1) LiDAR -> camera (nuScenes) sparse depth
            u, v, z = project_lidar_to_cam_sparse(
                nusc,
                lidar_sd,
                cam_sd,
                max_range=args.max_range,
                z_min=args.min_depth,
                z_max=args.max_depth,
            )

            if u.size == 0:
                continue

            # 2) Back-project using SfM pose into SfM world
            cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
            K = np.array(cam_cs["camera_intrinsic"], dtype=np.float32)

            img = cv2.imread(
                os.path.join(nusc.dataroot, cam_sd["filename"]), cv2.IMREAD_COLOR
            )
            if img is None:
                continue

            X_w, C = backproject_uvz_to_world_sfm(
                u, v, z, K, pose["T"], img
            )
            if X_w.shape[0] == 0:
                continue

            all_xyz.append(X_w)
            all_rgb.append(C)

    if len(all_xyz) == 0:
        print("No points produced. Exiting.")
        return

    xyz = np.concatenate(all_xyz, axis=0).astype(np.float32)
    rgb = np.concatenate(all_rgb, axis=0).astype(np.uint8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.out.lower().endswith(".ply"):
        write_ascii_ply_xyz_rgb(args.out, xyz, rgb)
        print(f"Saved PLY: {args.out}  points={xyz.shape[0]}")
    elif args.out.lower().endswith(".npy"):
        np.save(args.out, dict(xyz=xyz, rgb=rgb))
        print(f"Saved NPY: {args.out}  points={xyz.shape[0]}")
    else:
        raise ValueError("Output filename must end with .ply or .npy")

    print("Done.")


if __name__ == "__main__":
    main()