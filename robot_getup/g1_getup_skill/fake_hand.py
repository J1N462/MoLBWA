#!/usr/bin/env python3
"""
실제 로봇에 단 '가짜 손' 치수로 학습용 URDF와 MuJoCo 모델의 손을 똑같이 바꾼다.
(원본은 그대로 두고 *_fakehand 파일을 새로 만든다)

좌표: wrist_yaw_link 기준. x = 팔뚝 방향(손끝 쪽 +), y = 좌우, z = 위아래.
      --view 로 MuJoCo 뷰어를 띄우면 주황색 반투명 상자로 보이니, 실제 손과 방향·크기가 맞는지 눈으로 확인할 것.

  python3 fake_hand.py \
      --urdf  <whole_body_tracking>/.../unitree_description/urdf/g1/main.urdf \
      --mjcf  ~/dev/unitree_mujoco/unitree_robots/g1/g1_29dof.xml \
      --size 0.12 0.08 0.03 --start 0.04 --mass 0.25 --view

  --size  X Y Z : 손 상자 크기 [m] (팔뚝 방향 길이, 폭, 두께)
  --start S     : wrist_yaw 관절 중심 ~ 손이 시작되는 면까지 거리 [m] (원래 고무 손은 약 0.04)
  --mass  M     : 손 하나의 무게 [kg] (저울로 측정)
  --offset Y Z  : 상자 중심의 y, z 어긋남 [m] (왼손 기준, 오른손은 y 부호 반대) 기본 0 0
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--urdf", required=True)
ap.add_argument("--mjcf", required=True, help="unitree_mujoco g1_29dof.xml (scene 아님)")
ap.add_argument("--size", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
ap.add_argument("--start", type=float, required=True)
ap.add_argument("--mass", type=float, required=True)
ap.add_argument("--offset", type=float, nargs=2, default=[0.0, 0.0], metavar=("Y", "Z"))
ap.add_argument("--rubber-mass", type=float, default=None,
                help="URDF에 rubber_hand 링크가 없을 때 원래 고무 손 무게 [kg]")
ap.add_argument("--view", action="store_true")
a = ap.parse_args()

sx, sy, sz = a.size
cx = a.start + sx / 2


def center(side):
    y = a.offset[0] * (1 if side == "left" else -1)
    return np.array([cx, y, a.offset[1]])


# ====================================================================== URDF
tree = ET.parse(a.urdf)
root = tree.getroot()
links = {l.get("name"): l for l in root.findall("link")}
rubber_mass = {}
for side in ("left", "right"):
    wl = links[f"{side}_wrist_yaw_link"]
    for c in wl.findall("collision"):
        wl.remove(c)
    col = ET.SubElement(wl, "collision")
    ET.SubElement(col, "origin", xyz=" ".join(f"{v:.4f}" for v in center(side)), rpy="0 0 0")
    geo = ET.SubElement(col, "geometry")
    ET.SubElement(geo, "box", size=f"{sx:.4f} {sy:.4f} {sz:.4f}")

    rl = links.get(f"{side}_rubber_hand")
    if rl is not None:
        for c in rl.findall("collision"):
            rl.remove(c)
        inert = rl.find("inertial")
        m_old = float(inert.find("mass").get("value"))
        rubber_mass[side] = m_old
        inert.find("mass").set("value", f"{a.mass:.5f}")
        # rubber_hand 링크 좌표 = wrist_yaw_link 좌표 - 고정 관절 위치 (회전 없다고 가정)
        fj = next((j for j in root.findall("joint") if j.find("child").get("link") == f"{side}_rubber_hand"), None)
        jo = fj.find("origin") if fj is not None else None
        j_xyz = np.array([float(v) for v in jo.get("xyz", "0 0 0").split()]) if jo is not None else np.zeros(3)
        if jo is not None and any(abs(float(v)) > 1e-6 for v in jo.get("rpy", "0 0 0").split()):
            print(f"[urdf] 경고: {side}_rubber_hand 고정 관절에 회전이 있음 — 관성 위치가 약간 어긋날 수 있음")
        org = inert.find("origin")
        if org is None:
            org = ET.SubElement(inert, "origin")
        org.set("xyz", " ".join(f"{v:.4f}" for v in center(side) - j_xyz))
        org.set("rpy", "0 0 0")
        # 상자 관성 (균일 밀도)
        I = a.mass / 12 * np.array([sy ** 2 + sz ** 2, sx ** 2 + sz ** 2, sx ** 2 + sy ** 2])
        it = inert.find("inertia")
        for k, v in zip(("ixx", "iyy", "izz"), I):
            it.set(k, f"{v:.7f}")
        for k in ("ixy", "ixz", "iyz"):
            it.set(k, "0")
        print(f"[urdf] {side}_rubber_hand 질량 {m_old:.3f} → {a.mass:.3f} kg, 충돌은 wrist_yaw_link의 상자로 대체")
    else:
        if a.rubber_mass is None:
            raise SystemExit("[urdf] rubber_hand 링크가 없음 → --rubber-mass 로 원래 고무 손 무게를 알려주세요")
        rubber_mass[side] = a.rubber_mass
        m = wl.find("inertial/mass")
        m.set("value", f"{float(m.get('value')) - a.rubber_mass + a.mass:.5f}")
        print(f"[urdf] {side}_wrist_yaw_link 질량에 {a.mass - a.rubber_mass:+.3f} kg 반영")
urdf_out = os.path.join(os.path.dirname(a.urdf), "main_fakehand.urdf")
tree.write(urdf_out)
print("[urdf] 저장:", urdf_out)

# ====================================================================== MJCF
mt = ET.parse(a.mjcf)
mroot = mt.getroot()
for side in ("left", "right"):
    body = next(b for b in mroot.iter("body") if b.get("name") == f"{side}_wrist_yaw_link")
    # 기존 충돌 geom 끄기 (보이는 것만 남김)
    for g in body.findall("geom"):
        if g.get("contype") != "0":
            g.set("contype", "0")
            g.set("conaffinity", "0")
            g.set("group", "1")
            g.set("density", "0")
    c = center(side)
    ET.SubElement(body, "geom", name=f"{side}_fake_hand", type="box",
                  size=" ".join(f"{v / 2:.4f}" for v in (sx, sy, sz)), pos=" ".join(f"{v:.4f}" for v in c),
                  rgba="1 0.5 0 0.5", density="0")
    # 질량: 원래 wrist_yaw(고무 손 포함)에서 고무 손 무게를 빼고 가짜 손 무게를 더함, 무게중심도 이동
    inert = body.find("inertial")
    m0 = float(inert.get("mass"))
    p0 = np.array([float(v) for v in inert.get("pos").split()])
    dm = a.mass - rubber_mass[side]
    m1 = m0 + dm
    p1 = (m0 * p0 + dm * c) / m1 if m1 > 0 else p0
    inert.set("mass", f"{m1:.6f}")
    inert.set("pos", " ".join(f"{v:.6f}" for v in p1))
    d0 = np.array([float(v) for v in inert.get("diaginertia").split()])
    inert.set("diaginertia", " ".join(f"{v:.8f}" for v in d0 * max(m1, 1e-6) / m0))
    print(f"[mjcf] {side}_wrist_yaw_link 질량 {m0:.3f} → {m1:.3f} kg, 손 상자 추가")
mjcf_out = a.mjcf.replace(".xml", "_fakehand.xml")
mt.write(mjcf_out)
scene_in = os.path.join(os.path.dirname(a.mjcf), "scene_29dof.xml")
scene_out = os.path.join(os.path.dirname(a.mjcf), "scene_29dof_fakehand.xml")
if os.path.exists(scene_in):
    s = open(scene_in).read().replace(os.path.basename(a.mjcf), os.path.basename(mjcf_out))
    open(scene_out, "w").write(s)
print("[mjcf] 저장:", mjcf_out, "/", scene_out)

if a.view:
    import mujoco
    import mujoco.viewer
    m = mujoco.MjModel.from_xml_path(scene_out)
    d = mujoco.MjData(m)
    d.qpos[2] = 0.8
    mujoco.mj_forward(m, d)
    print("주황색 상자 = 새 손 충돌 모양. 회색 = 원래 고무 손(보이기만). 창을 닫으면 종료")
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            v.sync()
