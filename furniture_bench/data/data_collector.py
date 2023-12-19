"""Define data collection class that rollout the environment, get action from the interface (e.g., teleoperation, automatic scripts), and save data."""
import time
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import gym
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from ipdb import set_trace as st

from furniture_bench.device.device_interface import DeviceInterface
from furniture_bench.data.collect_enum import CollectEnum
from furniture_bench.config import config
from furniture_bench.perception.image_utils import resize, resize_crop
from furniture_bench.envs.initialization_mode import Randomness


class DataCollector:
    """Demonstration collection class.
    `pkl` files have resized images while `mp4` / `png` files save raw camera inputs.
    """

    def __init__(
        self,
        is_sim: bool,
        data_path: str,
        device_interface: DeviceInterface,
        furniture: str,
        headless: bool,
        draw_marker: bool,
        manual_label: bool,
        scripted: bool,
        randomness: Randomness.LOW,
        gpu_id: int = 0,
        pkl_only: bool = False,
        save_failure: bool = False,
        num_demos: int = 100,
        resize_img_after_sim: bool = True,
        small_sim_img_size: bool = False,
        verbose: bool = True,
        show_pbar: bool = False,
        obs_type: str = "state",
        encoder_type: str = "vip",
    ):
        """
        Args:
            is_sim (bool): Whether to use simulator or real world environment.
            data_path (str): Path to save data.
            device_interface (DeviceInterface): Keyboard and/or Oculus interface.
            furniture (str): Name of the furniture.
            headless (bool): Whether to use headless mode.
            draw_marker (bool): Whether to draw AprilTag marker.
            manual_label (bool): Whether to manually label the reward.
            scripted (bool): Whether to use scripted function for getting action.
            randomness (str): Initialization randomness level.
            gpu_id (int): GPU ID.
            pkl_only (bool): Whether to save only `pkl` files (i.e., exclude *.mp4 and *.png).
            save_failure (bool): Whether to save failure trajectories.
            num_demos (int): The maximum number of demonstrations to collect in this run. Internal loop will be terminated when this number is reached.
        """
        if is_sim:
            sim_type = dict(
                state="FurnitureSimState-v0",
                full="FurnitureSimFull-v0",
                image="FurnitureSimFull-v0",
                feature="FurnitureSimImageFeature-v0",
            )[obs_type]

            kwargs = dict(
                furniture=furniture,
                max_env_steps=600 if scripted else 3000,
                headless=headless,
                num_envs=1,  # Only support 1 for now.
                manual_done=False if scripted else True,
                randomness=randomness,
                compute_device_id=gpu_id,
                graphics_device_id=gpu_id,
                resize_img=small_sim_img_size,
            )
            if obs_type != "feature":
                kwargs.update(
                    resize_img=small_sim_img_size,
                    np_step_out=False,  # Always output Tensor in this setting. Will change to numpy in this code.
                    channel_first=False,
                )
            if obs_type == "feature":
                kwargs.update(
                    encoder_type=encoder_type,
                )

            self.env = gym.make(sim_type, **kwargs)
        else:
            if randomness == "med":
                randomness = Randomness.MEDIUM_COLLECT
            elif randomness == "high":
                randomness = Randomness.HIGH_COLLECT

            self.env = gym.make(
                "FurnitureBench-v0",
                furniture=furniture,
                resize_img=False,
                manual_done=True,
                with_display=not headless,
                draw_marker=draw_marker,
                randomness=randomness,
            )

        self.is_sim = is_sim
        self.data_path = Path(data_path)
        self.device_interface = device_interface
        self.headless = headless
        self.manual_label = manual_label
        self.furniture = furniture
        self.num_demos = num_demos
        self.scripted = scripted

        self.traj_counter = 0
        self.num_success = 0
        self.num_fail = 0

        self.pkl_only = pkl_only
        self.save_failure = save_failure
        self.resize_img_after_sim = resize_img_after_sim

        self.verbose = verbose
        self.pbar = None if not show_pbar else tqdm(total=self.num_demos)
        self.obs_type = obs_type

        self._reset_collector_buffer()

    def _squeeze_and_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        v = v.squeeze()
        return v

    def _set_dictionary(self, to, from_):
        if self.obs_type in ["full", "image"]:
            to["color_image1"] = from_["color_image1"]
            to["color_image2"] = from_["color_image2"]

            if self.resize_img_after_sim:
                to["color_image1"] = resize(to["color_image1"])
                to["color_image2"] = resize_crop(to["color_image2"])

            to["image_size"] = to["color_image2"].shape[:2]

        if self.obs_type in ["state", "full"]:
            to["parts_poses"] = from_["parts_poses"]

        if self.obs_type == "feature":
            to["feature1"] = from_["image1"]
            to["feature2"] = from_["image2"]

        to["robot_state"] = from_["robot_state"]

    def collect(self):
        self.verbose_print("[data collection] Start collecting the data!")

        obs = self.reset()
        done = False

        while self.num_success < self.num_demos:
            # Get an action.
            if self.scripted:
                action, skill_complete = self.env.get_assembly_action()
                collect_enum = CollectEnum.DONE_FALSE
            else:
                action, collect_enum = self.device_interface.get_action()
                skill_complete = int(collect_enum == CollectEnum.SKILL)
                if skill_complete == 1:
                    self.skill_set.append(skill_complete)

            if collect_enum == CollectEnum.TERMINATE:
                self.verbose_print("Terminate the program.")
                break

            # An episode is done.
            if done or collect_enum in [CollectEnum.SUCCESS, CollectEnum.FAIL]:
                print("done", done, "enumn", collect_enum)
                if self.is_sim:
                    # Convert it to numpy.
                    for k, v in next_obs.items():
                        if isinstance(v, dict):
                            for k1, v1 in v.items():
                                v[k1] = self._squeeze_and_numpy(v1)
                        else:
                            next_obs[k] = self._squeeze_and_numpy(v)

                self.org_obs.append(next_obs)

                n_ob = {}

                self._set_dictionary(to=n_ob, from_=next_obs)

                self.obs.append(n_ob)

                if done and not self.env.furnitures[0].all_assembled():
                    if self.save_failure:
                        self.verbose_print("Saving failure trajectory.")
                        collect_enum = CollectEnum.FAIL
                        obs = self.save_and_reset(collect_enum, {})
                    else:
                        self.verbose_print("Failed to assemble the furniture, reset without saving.")
                        obs = self.reset()
                        collect_enum = CollectEnum.SUCCESS
                    self.num_fail += 1
                else:
                    if done:
                        collect_enum = CollectEnum.SUCCESS

                    obs = self.save_and_reset(collect_enum, {})
                    self.num_success += 1
                    self.update_pbar()

                self.traj_counter += 1
                self.verbose_print(f"Success: {self.num_success}, Fail: {self.num_fail}")
                done = False
                continue

            # Execute action.
            next_obs, rew, done, info = self.env.step(action)

            if rew == 1:
                self.last_reward_idx = len(self.acts)

            # Label reward.
            if collect_enum == CollectEnum.REWARD:
                rew = self.env.furniture.manual_assemble_label(self.device_interface.rew_key)
                if rew == 0:
                    # Correction the label.
                    self.rews[self.last_reward_idx] = 0
                    rew = 1

            # Error handling.
            if not info["obs_success"]:
                self.verbose_print("Getting observation failed, save trajectory.")
                # Pop the last reward and action so that obs has length plus 1 then those of actions and rewards.
                self.rews.pop()
                self.acts.pop()
                obs = self.save_and_reset(CollectEnum.FAIL, info)
                continue

            # Logging a step.
            self.step_counter += 1
            self.verbose_print(
                f"{[self.step_counter]} assembled: {self.env.furniture.assembled_set} num assembled: {len(self.env.furniture.assembled_set)} Skill: {len(self.skill_set)}"
            )

            # Store a transition.
            if info["action_success"]:
                if self.is_sim:
                    for k, v in obs.items():
                        if isinstance(v, dict):
                            for k1, v1 in v.items():
                                v[k1] = self._squeeze_and_numpy(v1)
                        else:
                            obs[k] = self._squeeze_and_numpy(v)
                    if isinstance(rew, torch.Tensor):
                        rew = float(rew.squeeze().cpu())

                self.org_obs.append(obs.copy())
                ob = {}

                self._set_dictionary(to=ob, from_=obs)

                self.obs.append(ob)
                if self.is_sim:
                    if isinstance(action, torch.Tensor):
                        action = action.squeeze().cpu().numpy()
                    else:
                        action = action.squeeze()
                self.acts.append(action)
                self.rews.append(rew)
                self.skills.append(skill_complete)
            obs = next_obs

        self.verbose_print(f"Collected {self.traj_counter} / {self.num_demos} successful trajectories!")

    def save_and_reset(self, collect_enum: CollectEnum, info):
        """Saves the collected data and reset the environment."""
        self.save(collect_enum, info)
        self.verbose_print(f"Saved {self.traj_counter} trajectories in this run.")
        return self.reset()

    def reset(self):
        obs = self.env.reset()
        self._reset_collector_buffer()

        self.verbose_print("Start collecting the data!")
        if not self.scripted:
            self.verbose_print("Press enter to start")
            while True:
                if input() == "":
                    break
            time.sleep(0.2)

        return obs

    def _reset_collector_buffer(self):
        self.obs = []
        self.org_obs = []
        self.acts = []
        self.rews = []
        self.skills = []
        self.step_counter = 0
        self.last_reward_idx = -1
        self.skill_set = []

    def save(self, collect_enum: CollectEnum, info):
        self.verbose_print(f"Length of trajectory: {len(self.obs)}, collect_enum: {collect_enum}")

        data_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        demo_path = self.data_path / data_name
        demo_path.mkdir(parents=True, exist_ok=True)

        # Color data paths.
        self.color_names = ["color_image1", "color_image2", "color_image3"]
        self.color_video_names = []
        for name in self.color_names:
            self.color_video_names.append(demo_path / f"{data_name}_{name}.mp4")

        # Depth data paths.
        self.depth_names = ["depth_image1", "depth_image2", "depth_image3"]
        self.depth_paths = []
        for name in self.depth_names:
            self.depth_paths.append(demo_path / f"{data_name}_{name}")

        # Save data.
        path = demo_path / f"{data_name}.pkl"
        with open(path, "wb") as f:
            # Save transitions with resized images.
            data = {}
            data["observations"] = self.obs
            data["actions"] = self.acts
            data["rewards"] = self.rews
            data["skills"] = self.skills
            data["success"] = collect_enum == CollectEnum.SUCCES
            data["furniture"] = self.furniture

            if "error" in info:
                data["error_description"] = info["error"].value
                data["error"] = True
            else:
                data["error"] = False
                data["error_description"] = ""

            if not self.is_sim:
                data["cam2_to_base"] = self.env.cam_to_base[2]
                data["cam3_to_base"] = self.env.cam_to_base[3]

                data["cam1_intr"] = self.env.cam_intrs[1]
                data["cam2_intr"] = self.env.cam_intrs[2]
                data["cam3_intr"] = self.env.cam_intrs[3]

            # Save raw color images in mp4.
            if not self.pkl_only:
                self.verbose_print("Start saving raw color images.")
                outs = []
                for n in self.color_video_names:
                    outs.append(
                        cv2.VideoWriter(
                            str(n),
                            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                            10,
                            config["camera"]["color_img_size"],
                        )
                    )
                for i, k in enumerate(self.color_names):
                    for obs in self.org_obs:
                        outs[i].write(cv2.cvtColor(obs[k], cv2.COLOR_RGB2BGR))
                    outs[i].release()

                # Save raw depth images in png.
                self.verbose_print("Start saving raw depth images.")
                for i, k in enumerate(self.depth_names):
                    self.depth_paths[i].mkdir(parents=True, exist_ok=True)
                    Parallel(n_jobs=8)(
                        delayed(cv2.imwrite)(
                            f"{self.depth_paths[i]}/{j:05}.png",
                            obs[k],
                            [int(cv2.IMWRITE_PNG_COMPRESSION), 5],
                        )
                        for j, obs in enumerate(self.org_obs)
                    )

            pickle.dump(data, f)
        self.verbose_print(f"Data saved at {path}")

    def verbose_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def update_pbar(self):
        if self.pbar is not None:
            self.pbar.update(1)

    def __del__(self):
        del self.env

        if self.device_interface is not None:
            self.device_interface.close()
