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

        def __init__(self):
            """Initialize Differential Inverse Kinematics Controller.

            Args:
            """
            super().__init__()
            self.ee_pos_desired = None
            self.ee_quat_desired = None
            self.ee_pos_error = None
            self.ee_rot_error = None

            self.pos_scalar = 3.0
            self.rot_scalar = 10.0

        def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            # Get states.
            joint_pos_current = state_dict["joint_positions"]

            # 6x7
            jacobian = state_dict["jacobian_diffik"]

            position_error = self.ee_pos_error
            rot_error = self.ee_rot_error

            ee_delta_axis_angle = torch.from_numpy(rot_error.as_rotvec()).float().to(position_error.device)

            dt = 1.0
            ee_pos_vel = position_error * self.pos_scalar / dt
            ee_rot_vel = ee_delta_axis_angle * self.rot_scalar / dt

            ee_velocity_desired = torch.cat((ee_pos_vel, ee_rot_vel), dim=-1)
            joint_vel_desired = torch.linalg.lstsq(jacobian, ee_velocity_desired).solution
            joint_pos_desired = joint_pos_current + joint_vel_desired*dt

            return {"joint_positions": joint_pos_desired}

        def reset(self):
            pass

    return DiffIKController(*args, **kwargs)

