#!/usr/bin/env python3
"""두 G1 MJCF의 질량 분포와, 기상 중 가장 아슬아슬한 쪼그린 자세에서의 무게중심 vs 발 위치를 비교.
   python3 compare_mass.py g1_getup.csv <모델A.xml> <모델B.xml> ..."""
import sys
import numpy as np
import mujoco

SDK = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
       "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
       "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
       "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
       "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
       "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
       "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
GROUPS = {"pelvis": ["pelvis"], "torso(+head)": ["torso", "waist", "head", "logo", "imu", "d435", "mid360"],
          "legs": ["hip", "knee", "ankle"], "arms": ["shoulder", "elbow"], "hands": ["wrist", "hand", "rubber"]}

ref = np.loadtxt(sys.argv[1], delimiter=",")
frames = {"누움(0s)": 0, "쪼그림 9.3s": int(9.3 * 50), "쪼그림 10s": 500, "섬(끝)": len(ref) - 1}

for path in sys.argv[2:]:
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    print(f"\n=== {path.split('/')[-1]}  총질량 {m.body_subtreemass[1]:.2f} kg")
    for g, keys in GROUPS.items():
        mass = sum(m.body_mass[b] for b in range(1, m.nbody) if any(k in m.body(b).name for k in keys))
        print(f"   {g:13s} {mass:6.2f} kg")
    qadr = [m.jnt_qposadr[m.joint(n).id] for n in SDK]
    foot_geoms = [g for g in range(m.ngeom) if "ankle_roll" in m.body(m.geom_bodyid[g]).name
                  and (m.geom_contype[g] or m.geom_conaffinity[g])]
    for name, k in frames.items():
        r = ref[k]
        d.qpos[:] = 0
        d.qpos[0:3] = r[0:3]
        d.qpos[3:7] = r[[6, 3, 4, 5]]
        d.qpos[qadr] = r[7:]
        mujoco.mj_forward(m, d)
        com = d.subtree_com[1]
        # 발 충돌 geom의 AABB 꼭짓점으로 발바닥 앞뒤 범위 (pelvis 진행방향 x 기준)
        xs = []
        for g in foot_geoms:
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            R = d.geom_xmat[g].reshape(3, 3)
            pts = (np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c) @ R.T + d.geom_xpos[g]
            low = pts[pts[:, 2] < pts[:, 2].min() + 0.01]
            xs += list(low[:, 0])
        heel, toe = min(xs), max(xs)
        print(f"   {name:10s} CoM x {com[0]:+.3f} z {com[2]:.3f} | 발 x {heel:+.3f}~{toe:+.3f} | "
              f"뒤꿈치 앞 여유 {com[0] - heel:+.3f} m")
