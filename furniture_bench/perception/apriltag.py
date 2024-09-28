import pyrealsense2 as rs
import numpy as np
import cv2
from dt_apriltags import Detector
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from ipdb import set_trace as bp


import meshcat
from rdt.common import mc_util

zmq_url = f"tcp://127.0.0.1:6001"
mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
mc_vis["scene"].delete()


class AprilTag:
    def __init__(self, tag_size):
        self.at_detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.5,
            debug=0,
        )
        self.tag_size = tag_size

        self.last_det_rots = {}
        self.last_det_pos = {}
        self.alpha = 0.250

        self.det_num = 0

    def filter_detections(self, dets, cam_num: int = 1):
        valid_dets = []
        # return dets
        for det in dets:

            tag_id = det.tag_id

            # Just skip the base tags
            if tag_id in [0, 1, 2, 3]:
                continue

            if (tag_id, cam_num) not in self.last_det_rots:
                valid_dets.append(det)
                continue

            last_rot = self.last_det_rots[(tag_id, cam_num)]

            relative_pose = np.linalg.inv(last_rot) @ det.pose_R
            rotvec = R.from_matrix(relative_pose).as_rotvec()
            ang_vel_deg = np.rad2deg(np.linalg.norm(rotvec))

            if ang_vel_deg > 20:
                # if True:
                # print(f"Ang vel: {ang_vel_deg}, Tag ID: {tag_id}, Cam: {cam_num}")
                continue

            last_pos = self.last_det_pos[(tag_id, cam_num)]
            relative_pos = last_pos - det.pose_t

            if np.linalg.norm(relative_pos) > 0.05:
                print(
                    f"Relative pos: {np.linalg.norm(relative_pos)}, Tag ID: {tag_id}, Cam: {cam_num}"
                )
                continue

            valid_dets.append(det)
        return valid_dets

    def update_last_det(self, dets, cam_num: int = 1):
        for det in dets:
            if (det.tag_id, cam_num) in self.last_det_rots:
                slerp = Slerp(
                    [0, 1],
                    R.from_matrix(
                        [
                            self.last_det_rots[(det.tag_id, cam_num)],
                            det.pose_R,
                        ]
                    ),
                )
                self.last_det_rots[(det.tag_id, cam_num)] = slerp(
                    [self.alpha]
                ).as_matrix()[0]

                self.last_det_pos[(det.tag_id, cam_num)] = (
                    1 - self.alpha
                ) * self.last_det_pos[(det.tag_id, cam_num)] + self.alpha * det.pose_t

            else:
                self.last_det_rots[(det.tag_id, cam_num)] = det.pose_R
                self.last_det_pos[(det.tag_id, cam_num)] = det.pose_t

    def detect(self, frame, intr_param, cam_num: int = 1):
        """Detect AprilTag.

        Args:
            frame: pyrealsense2.frame or Gray-scale image to detect AprilTag.
            intr_param: Camera intrinsics format of [fx, fy, ppx, ppy].
        Returns:
            Detected tags.
        """
        if isinstance(frame, rs.frame):
            frame = np.asanyarray(frame.get_data())
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        detections = self.at_detector.detect(frame, True, intr_param, self.tag_size)
        if self.det_num > 20:
            valid_detections = self.filter_detections(detections, cam_num)
        else:
            valid_detections = detections

        self.update_last_det(detections, cam_num)
        self.det_num += 1

        return [detection for detection in valid_detections if detection.hamming < 2]

    def detect_id(self, frame, intr_param, cam_num: int = 1):
        detections = self.detect(frame, intr_param, cam_num=cam_num)
        # Make it as a dictionary which the keys are tag_id.
        return {detection.tag_id: detection for detection in detections}
