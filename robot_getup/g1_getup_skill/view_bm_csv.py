#!/usr/bin/env python3
"""BeyondMimic reference CSV (x,y,z,qx,qy,qz,qw,29 joints)를 MuJoCo 뷰어로 재생.
   python3 view_bm_csv.py ~/g1_data/g1_getup.csv --model ~/unitree_mujoco/unitree_robots/g1/scene_29dof.xml
"""
import argparse, time
import numpy as np
import mujoco, mujoco.viewer

JOINTS = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
          "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
          "waist_yaw", "waist_roll", "waist_pitch",
          "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
          "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
          "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
          "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--model", required=True)
ap.add_argument("--fps", type=float, default=50)
ap.add_argument("--speed", type=float, default=1.0)
a = ap.parse_args()

M = np.loadtxt(a.csv, delimiter=",")
m = mujoco.MjModel.from_xml_path(a.model)
d = mujoco.MjData(m)
qadr = [m.jnt_qposadr[m.joint(j + "_joint").id] for j in JOINTS]

with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        t0 = time.time()
        for k, row in enumerate(M):
            if not v.is_running():
                break
            d.qpos[0:3] = row[0:3]
            d.qpos[3:7] = row[[6, 3, 4, 5]]          # xyzw → wxyz
            d.qpos[qadr] = row[7:]
            mujoco.mj_forward(m, d)
            v.sync()
            while time.time() - t0 < k / a.fps / a.speed:
                time.sleep(0.001)
        time.sleep(1.0)
