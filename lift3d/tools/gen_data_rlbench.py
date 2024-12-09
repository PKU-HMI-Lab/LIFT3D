import argparse
import copy
import os
import pathlib
import pickle
import sys

import numpy as np
import tqdm
import zarr
from numcodecs import MsgPack
from rlbench.backend.observation import Observation
from rlbench.demo import keypoint_discovery
from rlbench.utils import get_stored_demos
from termcolor import colored

from lift3d.dataset import RLBenchDataset
from lift3d.envs import RLBenchEnv, RLBenchObservationConfig
from lift3d.helpers.common import (
    Logger,
    save_depth_image,
    save_point_cloud_ply,
    save_rgb_image,
    save_video_imageio,
)
from lift3d.helpers.graphics import EEpose, PointCloud, Quaternion


def action_extracter(data_curr, data_next, rotation_representation):
    arm_action = EEpose.pose_delta_7DoF(
        pose1_7d=data_curr.gripper_pose,
        pose2_7d=data_next.gripper_pose,
    )

    if rotation_representation == "quaternion":
        x, y, z, qx, qy, qz, qw = arm_action
        arm_action = np.concatenate(
            [
                np.array([x, y, z]),
                Quaternion.ensure_positive_real_part(
                    np.array([qx, qy, qz, qw]), scalar_first=False
                ),
            ],
            axis=0,
        )
    elif rotation_representation == "euler":
        arm_action = EEpose.pose_7DoF_to_6DoF(
            arm_action, scalar_first=False, degrees=False
        )
    else:
        raise ValueError("Rotation representation should be either quaternion or euler")

    gripper_action = data_next.gripper_open
    action = np.append(arm_action, gripper_action)
    return action


def robot_state_extracter(data: Observation):
    arm_joint_state = data.joint_positions
    arm_pose_state = data.gripper_pose
    x, y, z, qx, qy, qz, qw = arm_pose_state
    arm_pose_state = np.concatenate(
        [
            np.array([x, y, z]),
            Quaternion.ensure_positive_real_part(
                np.array([qx, qy, qz, qw]), scalar_first=False
            ),
        ],
        axis=0,
    )
    gripper_state = data.gripper_open
    robot_state = np.concatenate(
        (arm_joint_state, arm_pose_state, np.array([gripper_state]))
    )
    return robot_state


def image_extracter(data: Observation):
    image = data.front_rgb
    return image


def point_cloud_extracter(
    data: Observation, point_cloud_camera_names: list, num_points: int
):

    def get_point_cloud_single_view(data: Observation, camera_name: str):
        if camera_name == "wrist":
            point_cloud = data.wrist_point_cloud
            image = data.wrist_rgb
        elif camera_name == "front":
            point_cloud = data.front_point_cloud
            image = data.front_rgb
        elif camera_name == "left_shoulder":
            point_cloud = data.left_shoulder_point_cloud
            image = data.left_shoulder_rgb
        elif camera_name == "right_shoulder":
            point_cloud = data.right_shoulder_point_cloud
            image = data.right_shoulder_rgb
        elif camera_name == "overhead":
            point_cloud = data.overhead_point_cloud
            image = data.overhead_rgb
        elif camera_name == "corner1":
            point_cloud = data.corner1_point_cloud
            image = data.corner1_rgb
        elif camera_name == "egocentric":
            point_cloud = data.egocentric_point_cloud
            image = data.egocentric_rgb
        else:
            raise ValueError(f"Invalid camera name: {camera_name}")

        point_cloud = np.concatenate((point_cloud, image), axis=-1)
        point_cloud = point_cloud.reshape(-1, 6)
        return point_cloud

    point_clouds = []
    for camera_name in point_cloud_camera_names:
        point_cloud = get_point_cloud_single_view(data, camera_name)
        point_clouds.append(point_cloud)
    point_cloud = np.concatenate(point_clouds, axis=0)

    # Crop the point cloud
    xmin, ymin, zmin, xmax, ymax, zmax = RLBenchEnv.POINT_CLOUD_BOUNDS["default"]
    min_bound = np.array([xmin, ymin, zmin])
    max_bound = np.array([xmax, ymax, zmax])
    mask = np.all(point_cloud[:, :3] > min_bound, axis=1)
    point_cloud = point_cloud[mask]
    mask = np.all(point_cloud[:, :3] < max_bound, axis=1)
    point_cloud = point_cloud[mask]

    # Sample the point cloud
    point_cloud = PointCloud.point_cloud_sampling(point_cloud, num_points, "fps")

    return point_cloud


def main(args):
    # Report the arguments
    Logger.log_info(
        f'Running {colored(pathlib.Path(__file__).absolute(), "red")} with arguments:'
    )
    Logger.log_info(f"RLBench data root: {args.rlbench_data_root}")
    Logger.log_info(f"Task name: {args.task_name}")
    Logger.log_info(f"Camera name: {args.camera_name}")
    Logger.log_info(f"Rotation representation: {args.rotation_representation}")
    Logger.log_info(f"Image size: {args.image_size}")
    Logger.log_info(f"Number of episodes: {args.num_episodes}")
    Logger.log_info(f"Only keypoints: {args.only_keypoints}")
    Logger.log_info(f"Interval: {args.interval}")
    Logger.log_info(f"Number of points: {args.num_points}")
    Logger.log_info(f"Save directory: {args.save_dir}")
    task_name = args.task_name
    if args.rotation_representation not in ["quaternion", "euler"]:
        raise ValueError("Rotation representation should be either quaternion or euler")
    if args.camera_name != "front":
        raise ValueError("Only front camera is used for fixed manipulation")

    camera_names = list(set(args.point_cloud_camera_names) | set([args.camera_name]))
    Logger.log_info(f"Active camera names: {camera_names}")
    Logger.print_seperator()

    obs_config = RLBenchObservationConfig.multi_view_config(
        camera_names=camera_names,
        image_size=(args.image_size, args.image_size),
    )

    # Make directories
    video_dir = os.path.join(
        args.save_dir, "visualized_data", "videos", args.task_name, args.camera_name
    )
    image_dir = os.path.join(
        args.save_dir, "visualized_data", "images", args.task_name, args.camera_name
    )
    point_cloud_dir = os.path.join(
        args.save_dir,
        "visualized_data",
        "point_clouds",
        args.task_name,
        args.camera_name,
    )
    depth_dir = os.path.join(
        args.save_dir, "visualized_data", "depths", args.task_name, args.camera_name
    )
    text_dir = os.path.join(
        args.save_dir, "visualized_data", "texts", args.task_name, args.camera_name
    )
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(point_cloud_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    # Convert source data to dataset
    total_count = 0
    img_arrays = []
    point_cloud_arrays = []
    robot_state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    texts = []

    dataset_root = pathlib.Path(args.rlbench_data_root).expanduser()
    task_data_dir = dataset_root / task_name
    num_variations = len(
        list(filter(lambda x: x.startswith("variation"), os.listdir(task_data_dir)))
    )
    Logger.log_info(
        f'Found {colored(num_variations, "red")} variations for task {colored(task_name, "red")}'
    )
    iterable = (
        tqdm.tqdm(range(args.num_episodes)) if args.quiet else range(args.num_episodes)
    )
    for i in iterable:
        demo = get_stored_demos(
            amount=1,
            image_paths=None,
            dataset_root=dataset_root,
            variation_number=i % num_variations,
            task_name=task_name,
            obs_config=obs_config,
            from_episode_number=i // num_variations,
        )[0]

        description_path = (
            task_data_dir
            / f"variation{i % num_variations}"
            / "variation_descriptions.pkl"
        )
        with open(description_path, "rb") as f:
            descriptions = pickle.load(f)
        description = max(descriptions, key=len)

        total_count_sub = 0
        img_arrays_sub = []
        point_cloud_arrays_sub = []
        robot_state_arrays_sub = []
        action_arrays_sub = []
        texts_sub = []

        if not args.quiet:
            Logger.log_info(f'episode {colored(i, "red")}')

        key_points = keypoint_discovery(demo)
        action_points = (
            [0] + key_points
            if args.only_keypoints
            else sorted(list(set(range(0, len(demo), args.interval)) | set(key_points)))
        )

        if not args.quiet:
            Logger.log_info(
                f'extracted {colored(len(action_points), "red")} action points with '
                f'{colored(len(key_points), "red")} key points from '
                f'{colored(len(demo), "red")} steps'
            )

        demo = [demo[i] for i in action_points]
        for j in range(1, len(demo)):
            total_count_sub += 1
            # action
            action = action_extracter(
                demo[j - 1], demo[j], args.rotation_representation
            )
            # robot state
            robot_state = robot_state_extracter(demo[j - 1])
            # record data
            img_arrays_sub.append(image_extracter(demo[j - 1]))
            point_cloud_arrays_sub.append(
                point_cloud_extracter(demo[j - 1], camera_names, args.num_points)
            )
            robot_state_arrays_sub.append(robot_state)
            action_arrays_sub.append(action)
            texts_sub.append(description)

        # save visualized data
        sample_video_array = np.stack(img_arrays_sub, axis=0)
        save_video_imageio(
            sample_video_array,
            os.path.join(video_dir, f"episode_{i}.mp4"),
            quiet=args.quiet,
        )
        save_rgb_image(
            img_arrays_sub[0],
            os.path.join(image_dir, f"episode_{i}.png"),
            quiet=args.quiet,
        )
        save_depth_image(
            demo[0].front_depth,
            os.path.join(depth_dir, f"episode_{i}.png"),
            quiet=args.quiet,
        )
        save_point_cloud_ply(
            point_cloud_arrays_sub[0],
            os.path.join(point_cloud_dir, f"episode_{i}.ply"),
            quiet=args.quiet,
        )
        with open(os.path.join(text_dir, f"{task_name}_episode_{i}.txt"), "w") as f:
            f.write(description)

        # merge episode data into dataset
        total_count += total_count_sub
        episode_ends_arrays.append(
            copy.deepcopy(total_count)
        )  # the index of the last step of the episode
        img_arrays.extend(copy.deepcopy(img_arrays_sub))
        point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
        robot_state_arrays.extend(copy.deepcopy(robot_state_arrays_sub))
        action_arrays.extend(copy.deepcopy(action_arrays_sub))
        texts.extend(copy.deepcopy(texts_sub))

        # release memory
        del (
            demo,
            key_points,
            action_points,
            img_arrays_sub,
            point_cloud_arrays_sub,
            robot_state_arrays_sub,
            action_arrays_sub,
            texts_sub,
        )

        if not args.quiet:
            Logger.log_info(
                f'Episode index: {colored(i, "red")}, '
                f'Episode end: {colored(total_count, "red")}, '
                f'Episode length: {colored(total_count_sub, "red")}'
            )
            Logger.print_seperator()

    # Merge data
    Logger.log_info("Merging episdoes sub-data to whole dataset...", end="", flush=True)
    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3:  # make channel last
        img_arrays = np.transpose(img_arrays, (0, 2, 3, 1))
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    robot_state_arrays = np.stack(robot_state_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)
    texts = np.array(texts, dtype=object)
    print("Done")

    # Save data
    Logger.log_info("Saving data to zarr file...", end="", flush=True)
    zarr_dir = pathlib.Path(args.save_dir) / f"{task_name}.zarr"
    zarr_root = zarr.group(zarr_dir)
    zarr_data = zarr_root.create_group("data", overwrite=True)
    zarr_meta = zarr_root.create_group("meta", overwrite=True)
    img_chunk_size = (
        100,
        img_arrays.shape[1],
        img_arrays.shape[2],
        img_arrays.shape[3],
    )
    point_cloud_chunk_size = (
        100,
        point_cloud_arrays.shape[1],
        point_cloud_arrays.shape[2],
    )
    robot_state_chunk_size = (100, robot_state_arrays.shape[1])
    action_chunk_size = (100, action_arrays.shape[1])
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_data.create_dataset(
        "images",
        data=img_arrays,
        chunks=img_chunk_size,
        dtype="uint8",
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "point_clouds",
        data=point_cloud_arrays,
        chunks=point_cloud_chunk_size,
        dtype="float32",
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "robot_states",
        data=robot_state_arrays,
        chunks=robot_state_chunk_size,
        dtype="float32",
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "actions",
        data=action_arrays,
        chunks=action_chunk_size,
        dtype="float32",
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends", data=episode_ends_arrays, dtype="int64", compressor=compressor
    )
    zarr_data.create_dataset(
        "texts", data=texts, dtype=object, compressor=compressor, object_codec=MsgPack()
    )
    print("Done")
    Logger.log_info(f"Dataset Info:\n{zarr_root.tree()}")
    Logger.print_seperator()
    del (
        img_arrays,
        point_cloud_arrays,
        robot_state_arrays,
        action_arrays,
        episode_ends_arrays,
        texts,
    )
    del zarr_root, zarr_data, zarr_meta
    Logger.log_info("Delete the data in memory")

    # validate the saved data
    dataset = RLBenchDataset(
        data_dir=zarr_dir,
        split="custom",
        custom_split_size=max(10, args.num_episodes // 10),
    )
    dataset.print_info()

    Logger.log_ok("All data saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlbench-data-root", type=str, default="~/Data/RLBench_224")
    parser.add_argument("--task-name", type=str, default="open_box")
    parser.add_argument("--camera-name", type=str, default="front")
    parser.add_argument(
        "--point-cloud-camera-names",
        type=str,
        nargs="+",
        default=["front", "overhead", "wrist", "left_shoulder", "right_shoulder"],
    )
    parser.add_argument(
        "--rotation-representation",
        type=str,
        default="quaternion",
        help="quaternion or euler",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=120)
    parser.add_argument("--only-keypoints", action="store_true")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(
            pathlib.Path(__file__).resolve().parent.parent.parent
            / "data"
            / "rlbench_i1_quaternion"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    main(parser.parse_args())
