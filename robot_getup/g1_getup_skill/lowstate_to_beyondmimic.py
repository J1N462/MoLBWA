#!/usr/bin/env python3
"""
G1 lowstate 로그 → BeyondMimic(whole_body_tracking) reference CSV

출력 CSV (헤더 없음, 프레임당 1줄):
  base_x, base_y, base_z, qx, qy, qz, qw, 29개 관절각 (unitree SDK 순서 = csv_to_npz.py joint_names 순서)

lowstate에는 base 위치가 없으므로 MuJoCo FK로 복원한다.
  z  : 매 프레임 가장 낮은 충돌 지점이 지면(+clearance)에 닿도록
  xy : 직전 프레임과 현재 프레임 모두 바닥에 닿아 있는 지점들이 미끄러지지 않도록 누적
  자세: IMU roll/pitch/yaw (시작 yaw를 0으로 정렬)

예)
  python3 lowstate_to_beyondmimic.py motor_20261019_232356.csv \
      --model ~/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
      --t-start 4.3 --t-end 23.0 --out g1_getup.csv
"""
import argparse
import numpy as np
import pandas as pd
import mujoco

JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def rpy_to_wxyz(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return np.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], axis=-1)


def smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, i], k, mode="valid")[: len(x)] for i in range(x.shape[1])], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", required=True, help="unitree_mujoco g1 scene_29dof.xml")
    ap.add_argument("--t-start", type=float, default=4.3, help="로그 시간 기준 시작[s] (기본: 기상 시작 0.5초 전)")
    ap.add_argument("--t-end", type=float, default=23.0, help="로그 시간 기준 끝[s] (기본: 서기 후 약 5.6초)")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--clearance", type=float, default=0.005, help="최저점과 지면 사이 여유[m]")
    ap.add_argument("--contact-tol", type=float, default=0.012, help="접촉 판정 높이[m]")
    ap.add_argument("--smooth", type=int, default=5, help="base 위치 이동평균 창(프레임)")
    ap.add_argument("--out", default="g1_getup.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    t_src = df["t"].to_numpy(float)
    t = np.arange(args.t_start, min(args.t_end, t_src[-1]) + 1e-9, 1.0 / args.fps)
    Q = np.stack([np.interp(t, t_src, df[f"q_{j}"].to_numpy(float)) for j in JOINTS], axis=1)
    rpy = np.stack([np.interp(t, t_src, np.unwrap(df[c].to_numpy(float))) for c in ("imu_roll", "imu_pitch", "imu_yaw")], 1)
    rpy[:, 2] -= rpy[0, 2]
    quat = rpy_to_wxyz(*rpy.T)
    print(f"{len(t)} frames @ {args.fps} Hz, {t[0]:.2f}~{t[-1]:.2f}s (로그 시간)")

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    jid = [m.joint(j + "_joint").id for j in JOINTS]
    qadr = np.array([m.jnt_qposadr[j] for j in jid])
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    n_clip = int(np.sum((Q < lo) | (Q > hi)))
    Q = np.clip(Q, lo, hi)
    if n_clip:
        print(f"관절 한계 밖 {n_clip}개 값을 모델 한계로 클립 (hip pitch/knee 끝단, 0.03 rad 이내)")

    geoms = [g for g in range(m.ngeom)
             if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE and (m.geom_contype[g] or m.geom_conaffinity[g])]

    def corners_world(k, xy):
        d.qpos[:] = 0
        d.qpos[0:2] = xy
        d.qpos[2] = 0.0
        d.qpos[3:7] = quat[k]
        d.qpos[qadr] = Q[k]
        mujoco.mj_kinematics(m, d)
        pts = []
        for g in geoms:
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            R = d.geom_xmat[g].reshape(3, 3)
            pts.append((CORNERS * h + c) @ R.T + d.geom_xpos[g])
        return np.concatenate(pts, 0)

    base = np.zeros((len(t), 3))
    prev = None
    n_contacts = []
    for k in range(len(t)):
        xy = base[k - 1, :2] if k else np.zeros(2)
        P = corners_world(k, xy)
        z_off = args.clearance - P[:, 2].min()
        P[:, 2] += z_off
        if prev is not None:
            mask = (P[:, 2] < args.clearance + args.contact_tol) & (prev[:, 2] < args.clearance + args.contact_tol)
            if mask.sum() >= 1:
                shift = (prev[mask, :2] - P[mask, :2]).mean(0)
                xy = xy + shift
                P[:, :2] += shift
            n_contacts.append(int(mask.sum()))
        base[k] = [xy[0], xy[1], z_off]
        prev = P

    # 위치 평활화 후 지면 관통 재보정
    base = smooth(base, args.smooth)
    pen = []
    for k in range(len(t)):
        P = corners_world(k, base[k, :2])
        zmin = P[:, 2].min() + base[k, 2]
        if zmin < args.clearance:
            base[k, 2] += args.clearance - zmin
        pen.append(zmin)

    # 슬립 점검: 양쪽 프레임 모두 접촉한 지점의 수평 이동
    slips = []
    prev = None
    for k in range(len(t)):
        P = corners_world(k, base[k, :2]); P[:, 2] += base[k, 2]
        if prev is not None:
            mask = (P[:, 2] < args.clearance + args.contact_tol) & (prev[:, 2] < args.clearance + args.contact_tol)
            if mask.any():
                slips.append(np.linalg.norm(P[mask, :2] - prev[mask, :2], axis=1).max())
        prev = P

    out = np.concatenate([base, quat[:, [1, 2, 3, 0]], Q], axis=1)  # quat → xyzw
    np.savetxt(args.out, out, delimiter=",", fmt="%.6f")

    print(f"base 이동: x {base[0,0]:+.3f} → {base[-1,0]:+.3f} m, y {base[0,1]:+.3f} → {base[-1,1]:+.3f} m")
    print(f"pelvis 높이: 시작 {base[0,2]:.3f} m, 최대 {base[:,2].max():.3f} m, 끝 {base[-1,2]:.3f} m")
    print(f"접촉 지점 프레임당 중앙값 {np.median(n_contacts):.0f}개, 최소 {min(n_contacts)}개")
    print(f"접촉 지점 슬립: 중앙값 {np.median(slips)*1000:.1f} mm/frame, 최대 {np.max(slips)*1000:.1f} mm/frame")
    print(f"저장: {args.out}  ({out.shape[0]} x {out.shape[1]}, input_fps={args.fps})")


if __name__ == "__main__":
    main()
