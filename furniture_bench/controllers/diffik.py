import math
from typing import Dict, List

import torch
from scipy.spatial.transform import Rotation as R

import furniture_bench.controllers.control_utils as C


def diffik_factory(real_robot=True, *args, **kwargs):
    if real_robot:
        import torchcontrol as toco

        base = toco.PolicyModule
    else:
        base = torch.nn.Module

    class DiffIKController(base):
        """Differential Inverse Kinematics Controller"""

        def __init__(
            self,
            kp: torch.Tensor,
            kv: torch.Tensor,
            ee_pos_current: torch.Tensor,
            ee_quat_current: torch.Tensor,
            init_joints: torch.Tensor,
            position_limits: torch.Tensor,
            mass_matrix_offset_val: List[float] = [0.2, 0.2, 0.2],
            max_dx: float = 0.005,
            controller_freq: int = 1000,
            policy_freq: int = 5,
            ramp_ratio: float = 1,
            joint_kp: float = 10.0,
        ):
            """Initialize Differential Inverse Kinematics Controller.

            Args:
                kp (torch.Tensor): positional gain for determining desired torques based upon the pos / ori errors.
                                    Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)
                kv (torch.Tensor): velocity gain for determining desired torques based on vel / ang vel errors.
                                    Can be either a scalar (same value for all action dims) or list (specific values for each dim).
                                    If kv is defined, damping is ignored.
                ee_pos_current (torch.Tensor): Current end-effector position.
                ee_quat_current (torch.Tensor): Current end-effector orientation.
                init_joints (torch.Tensor): Initial joint position (for nullspace).
                position_limits (torch.Tensor): Limits (m) below and above which the magnitude
                                                of a calculated goal eef position will be clipped. Can be either be a 2-list (same min/max value for all
                                                cartesian dims), or a 2-list of list (specific min/max values for each dim)
                mass_matrix_offset_val (list): 3f list of offsets to add to the mass matrix diagonal's last three elements.
                                                Used for real robots to adjust for high friction at end joints.
                max_dx (float): Maximum delta of positional movement in interpolation.
                control_freq (int): Frequency of control loop.
                policy_freq (int): Frequency at which actions from the robot policy are fed into this controller
                ramp_ratio (float): Ratio of control_freq / policy_freq. Used to determine how many steps to take in the interpolator.
                joint_kp (float): Proportional gain for joint position control.
            """
            super().__init__()
            # limits
            self.position_limits = position_limits
            self.kp = kp
            self.kv = kv
            self.init_joints = init_joints

            # self.ee_pos_desired = torch.nn.Parameter(ee_pos_current)
            # self.ee_quat_desired = torch.nn.Parameter(ee_quat_current)
            self.ee_pos_desired = None
            self.ee_quat_desired = None
            self.ee_pos_error = None
            self.ee_rot_error = None

            self.pos_scalar = 3.0
            self.rot_scalar = 10.0

            # self.mass_matrix = torch.zeros((7, 7))
            self.mass_matrix_offset_val = mass_matrix_offset_val
            self.mass_matrix_offset_idx = torch.tensor([[4, 4], [5, 5], [6, 6]])

            self.repeated_torques_counter = 0
            self.num_repeated_torques = 3
            self.prev_torques = torch.zeros((7,))

            # Interpolator pos, ori
            self.max_dx = max_dx  # Maximum allowed change per interpolator step
            self.total_steps = math.floor(
                ramp_ratio * float(controller_freq) / float(policy_freq)
            )  # Total num steps per interpolator action
            # Save previous goal
            self.goal_pos = ee_pos_current.clone()
            self.prev_goal_pos = ee_pos_current.clone()
            self.step_num_pos = 1

            self.fraction = 0.5
            self.goal_ori = ee_quat_current.clone()
            self.prev_goal_ori = ee_quat_current.clone()
            self.step_num_ori = 1
            self.prev_interp_pos = ee_pos_current.clone()
            self.prev_interp_ori = ee_quat_current.clone()

            self.joint_kp = joint_kp

        def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            # self.repeated_torques_counter = (self.repeated_torques_counter + 1) % self.num_repeated_torques
            # if self.repeated_torques_counter != 1:
            #     return {"joint_torques": self.prev_torques}

            # Get states.
            joint_pos_current = state_dict["joint_positions"]
            joint_vel_current = state_dict["joint_velocities"]

            ee_pose = state_dict["ee_pose"].reshape(4, 4).t().contiguous()
            ee_pos, ee_quat = C.mat2pose(ee_pose)
            ee_pos = ee_pos.to(ee_pose.device)
            ee_quat = ee_quat.to(ee_pose.device)

            # 6x7
            jacobian = state_dict["jacobian_diffik"]

            if False:
                # print(f'Pos: {ee_pos}, Pos (des): {self.ee_pos_desired}')
                # print(f'Ori: {ee_quat}, Ori (des): {self.ee_quat_desired}')

                # ee_pos_desired = ee_pos
                ee_pos_desired = self.ee_pos_desired
                # ee_pos_desired = C.set_goal_position(self.position_limits, self.ee_pos_desired)
                ee_quat_desired = self.ee_quat_desired  

                # Setting goal_pos, goal_ori.
                # self.set_goal(ee_pos_desired, ee_quat_desired)
                # ee_pos_desired = self.get_interpolated_goal_pos()
                # ee_quat_desired = self.get_interpolated_goal_ori()

                # Calculate desired force, torque at ee using control law and error.
                position_error = ee_pos_desired - ee_pos

            position_error = self.ee_pos_error
            rot_error = self.ee_rot_error

            # goal_ori_mat = C.quat2mat(ee_quat_desired).to(ee_quat_desired.device)
            # ee_ori_mat = C.quat2mat(ee_quat).to(ee_quat.device)
            # ee_delta_quat = C.mat2quat(goal_ori_mat @ torch.inverse(ee_ori_mat))
            # ee_delta_quat = C.orientation_error_quat_flat(ee_quat_desired, ee_quat)
            # ee_delta_axis_angle = C.quaternion_to_axis_angle(ee_delta_quat) 

            # print(f'Pos error: {position_error}')
            # print(f'Ori error: {ee_delta_axis_angle}')

            ee_delta_axis_angle = torch.from_numpy(rot_error.as_rotvec()).float().to(position_error.device)

            # dt = 0.01
            dt = 1.0
            ee_pos_vel = position_error * self.pos_scalar / dt
            ee_rot_vel = ee_delta_axis_angle * self.rot_scalar / dt
            # ee_rot_vel = torch.zeros(3).float().to(ee_quat.device)
            # print(f'position_error: {position_error}')
            # print(f'ee_delta_axis_angle: {ee_delta_axis_angle}')
            # print(f'ee_rot_vel: {ee_rot_vel}')

            ee_velocity_desired = torch.cat((ee_pos_vel, ee_rot_vel), dim=-1)
            # print(f'ee_velocity_desired: {ee_velocity_desired.shape}')
            # print(f'Here to confirm we can compute IK')
            # from IPython import embed; embed()
            joint_vel_desired = torch.linalg.lstsq(jacobian, ee_velocity_desired).solution
            # print(f'ee_velocity_desired: {ee_velocity_desired}')
            # print(f'joint_vel_desired: {joint_vel_desired}')
            joint_pos_desired = joint_pos_current + joint_vel_desired*dt

            # print(f'joint_pos_desired: {joint_pos_desired}')
            # print(f'joint_pos_current: {joint_pos_current}')
            
            if False:
                torque_feedback = (
                    torch.multiply(self.kp, (joint_pos_desired - joint_pos_current)) - 
                    torch.multiply(self.kv, joint_vel_current)
                )

                # torque_feedback = torch.multiply(self.kp, (joint_pos_desired - joint_pos_current))

                # torque_feedforward = self.invdyn(joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current))
                torque_feedforward = torch.zeros_like(torque_feedback)
                torques = torque_feedback + torque_feedforward
                # torques = torch.zeros(7).float().to(ee_quat.device)
                self.prev_torques = torques
            # self._torque_offset(ee_pos, goal_pos, torques)
        
            # print(f'torques: {torques}')

            # return {"joint_torques": torques}
            return {"joint_positions": joint_pos_desired}

        def set_goal(self, goal_pos, goal_ori):
            if (
                not torch.isclose(goal_pos, self.goal_pos).all()
                or not torch.isclose(goal_ori, self.goal_ori).all()
            ):
                self.prev_goal_pos = self.goal_pos.clone()
                self.goal_pos = goal_pos.clone()
                self.step_num_pos = 1

                self.prev_goal_ori = self.goal_ori.clone()
                self.goal_ori = goal_ori.clone()
                self.step_num_ori = 1
            elif (
                self.step_num_pos >= self.total_steps
                or self.step_num_ori >= self.total_steps
            ):
                self.prev_goal_pos = self.prev_interp_pos.clone()
                self.goal_pos = goal_pos.clone()
                self.step_num_pos = 1

                self.prev_goal_ori = self.prev_interp_ori.clone()
                self.goal_ori = goal_ori.clone()
                self.step_num_ori = 1

        def get_interpolated_goal_pos(self) -> torch.Tensor:
            # Calculate the desired next step based on remaining interpolation steps and increment step if necessary
            dx = (self.goal_pos - self.prev_goal_pos) / (self.total_steps)
            # Check if dx is greater than max value; if it is; clamp and notify user
            if torch.any(abs(dx) > self.max_dx):
                dx = torch.clip(dx, -self.max_dx, self.max_dx)

            interp_goal = self.prev_goal_pos + (self.step_num_pos + 1) * dx
            self.step_num_pos += 1
            self.prev_interp_pos = interp_goal
            return interp_goal

        def get_interpolated_goal_ori(self):
            """Get interpolated orientation using slerp."""
            interp_fraction = (self.step_num_ori / self.total_steps) * self.fraction
            interp_goal = C.quat_slerp(
                self.prev_goal_ori, self.goal_ori, fraction=interp_fraction
            )
            self.step_num_ori += 1
            self.prev_interp_ori = interp_goal

            return interp_goal

        def _torque_offset(self, ee_pos, goal_pos, torques):
            """Torque offset to prevent robot from getting stuck when reached too far."""
            if (
                ee_pos[0] >= self.position_limits[0][1]
                and goal_pos[0] - ee_pos[0] <= -self.max_dx
            ):
                torques[1] -= 2.0
                torques[3] -= 2.0

        def reset(self):
            self.repeated_torques_counter = 0

    return DiffIKController(*args, **kwargs)

