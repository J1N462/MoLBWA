#!/usr/bin/env python3
"""
MuJoCo sim2sim: 누운 자세 → GetupSkill → 안정 판정 → 인계.
인계 뒤에는 팀원 균형 제어 대신 '임시 대역'(현재 자세를 높은 gain으로 잡는 PD)을 붙여서, 인계 순간에 튀는지만 본다.

  python3 sim_getup.py --policy getup.onnx --start-csv g1_getup.csv \
      --model ~/unitree_mujoco/unitree_robots/g1/scene_29dof.xml --viewer
  python3 sim_getup.py ... --friction 0.6 --kp-scale 0.9 --seed 1        # 강건성
  python3 sim_getup.py ... --balance my_balance:Controller              # 팀원 코드로 인계 (아래 인터페이스)

팀원 균형 제어 인터페이스(같은 형식이면 sim/실기 모두 그대로 꽂힘):
    class Controller:
        def start(self, state, handoff_info): ...
        def step(self, state) -> {"q","dq","kp","kd"}   # 50 Hz, SDK 순서 29개
"""
import argparse
import importlib
import time

import numpy as np
import mujoco

from getup_skill import SDK_JOINTS, POLICY_DT, GetupSkill, smoothstep

A5020, A7520_14, A7520_22, A4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425
ARMATURE = {"hip_pitch": A7520_14, "hip_roll": A7520_22, "hip_yaw": A7520_14, "knee": A7520_22,
            "ankle": 2 * A5020, "waist_roll": 2 * A5020, "waist_pitch": 2 * A5020, "waist_yaw": A7520_14,
            "shoulder": A5020, "elbow": A5020, "wrist_roll": A5020, "wrist_pitch": A4010, "wrist_yaw": A4010}
EFFORT = {"hip_pitch": 88, "hip_roll": 139, "hip_yaw": 88, "knee": 139, "ankle": 50, "waist_roll": 50,
          "waist_pitch": 50, "waist_yaw": 88, "shoulder": 25, "elbow": 25, "wrist_roll": 25,
          "wrist_pitch": 5, "wrist_yaw": 5}


def lookup(table, name):
    return next(v for k, v in table.items() if k in name)


class StandInBalance:
    """팀원 균형 제어 자리를 임시로 채우는 대역: 인계 순간 자세를 높은 gain으로 잡음 (0.3초 gain 블렌딩).
    능동 균형이 아니므로 오래 서 있는 성능은 의미 없음 — 인계 순간 충격 확인용."""

    KP = {"hip": 200, "knee": 250, "ankle": 300, "waist": 200, "shoulder": 40, "elbow": 40, "wrist": 15}
    KD = {"hip": 5, "knee": 6, "ankle": 6, "waist": 5, "shoulder": 1.5, "elbow": 1.5, "wrist": 0.5}

    def start(self, state, info):
        self.q_hold = info["q_measured"].copy()
        self.kp0, self.kd0 = info["kp"], info["kd"]
        self.kp1 = np.array([lookup(self.KP, n) for n in SDK_JOINTS], float)
        self.kd1 = np.array([lookup(self.KD, n) for n in SDK_JOINTS], float)
        self.k = 0

    def step(self, state):
        self.k += 1
        w = smoothstep(self.k * POLICY_DT / 0.3)
        return dict(q=self.q_hold, dq=np.zeros(29), kp=(1 - w) * self.kp0 + w * self.kp1,
                    kd=(1 - w) * self.kd0 + w * self.kd1)


def model_report(m, jid, dadr):
    """학습(Isaac)과 다를 수 있는 동역학 설정 요약."""
    names = [m.joint(j).name for j in jid]
    print(f"[model] timestep {m.opt.timestep}s, integrator {mujoco.mjtIntegrator(m.opt.integrator).name}, "
          f"cone {mujoco.mjtCone(m.opt.cone).name}, 총질량 {m.body_subtreemass[1]:.2f} kg")
    print(f"[model] actuator {m.nu}개: " + ", ".join(sorted({
        f"{mujoco.mjtGain(m.actuator_gaintype[a]).name}/{mujoco.mjtBias(m.actuator_biastype[a]).name}" for a in range(m.nu)})))
    for key in ("hip_pitch", "knee", "ankle_pitch", "waist_yaw", "elbow", "wrist_yaw"):
        i = next(k for k, n in enumerate(names) if key in n)
        da = dadr[i]
        print(f"[model] {names[i]:26s} damping {m.dof_damping[da]:.3f}  frictionloss {m.dof_frictionloss[da]:.3f}  "
              f"armature {m.dof_armature[da]:.4f}  range {np.round(m.jnt_range[jid[i]], 2)}")
    col = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    kinds = {}
    for g in col:
        k = mujoco.mjtGeom(m.geom_type[g]).name.replace("mjGEOM_", "")
        kinds[k] = kinds.get(k, 0) + 1
    print(f"[model] 충돌 geom {len(col)}개: {kinds}")


def load_model(path, no_self_collision=False):
    """아무 G1 MJCF나 로드. 바닥이 없으면 추가. 토크는 qfrc_applied로 넣고, 모델에 있는 actuator는 힘이 0이 되도록 무력화."""
    spec = mujoco.MjSpec.from_file(path)
    m = spec.compile()
    if not any(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in range(m.ngeom)):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], friction=[1.0, 0.005, 0.0001])
        m = spec.compile()
        print("[sim] 모델에 바닥이 없어 추가함")
    if no_self_collision:
        for g in range(m.ngeom):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                m.geom_contype[g], m.geom_conaffinity[g] = 1, 1
            elif m.geom_contype[g] or m.geom_conaffinity[g]:
                m.geom_contype[g], m.geom_conaffinity[g] = 2, 1        # 로봇끼리는 충돌 안 함, 바닥과만
        print("[sim] 자가충돌 끔 (로봇-바닥 접촉만)")
    if m.nu:
        # position/velocity 서보형 actuator가 ctrl=0을 목표로 관절을 끌어당기지 않도록 이득을 0으로
        m.actuator_gainprm[:] = 0
        m.actuator_biasprm[:] = 0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--start-csv", required=True, help="g1_getup.csv — 시뮬 시작 자세(누움)로만 사용")
    ap.add_argument("--model", required=True)
    ap.add_argument("--balance", default=None, help="팀원 균형 제어 'module:Class' (없으면 임시 대역)")
    ap.add_argument("--after", type=float, default=5.0, help="인계 후 관찰 시간[s]")
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--no-self-collision", action="store_true", help="진단용: 로봇 자가충돌 끄기")
    ap.add_argument("--diag", action="store_true", help="0.5초마다 기준 대비 가장 어긋난 관절 출력")
    ap.add_argument("--match-train", action="store_true",
                    help="관절 damping·frictionloss를 0으로 (학습 모델은 armature만 있음)")
    ap.add_argument("--inspect", action="store_true", help="모델 동역학 설정만 출력하고 종료")
    ap.add_argument("--elliptic", action="store_true",
                    help="마찰 원뿔을 elliptic + impratio 10으로 (미끄러짐이 적은, PhysX에 가까운 접촉)")
    ap.add_argument("--record", default=None, help="매 틱 q_sim / q_des / q_ref 저장할 CSV 경로")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    m = load_model(a.model, a.no_self_collision)
    d = mujoco.MjData(m)
    if a.friction is not None:
        m.geom_friction[:, 0] = a.friction
    jid = [m.joint(n).id for n in SDK_JOINTS]
    qadr = np.array([m.jnt_qposadr[j] for j in jid])
    dadr = np.array([m.jnt_dofadr[j] for j in jid])
    for n, da in zip(SDK_JOINTS, dadr):
        m.dof_armature[da] = lookup(ARMATURE, n)
    effort = np.array([lookup(EFFORT, n) for n in SDK_JOINTS], float)
    pelvis = m.body("pelvis").id
    if a.elliptic:
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        m.opt.impratio = 10.0
        print("[sim] 접촉: elliptic cone, impratio 10")
    if a.match_train:
        m.dof_damping[dadr] = 0.0
        m.dof_frictionloss[dadr] = 0.0
        print("[sim] 관절 damping·frictionloss = 0 (학습 조건)")
    model_report(m, jid, dadr)
    if a.inspect:
        return

    t_sim = 0.0
    skill = GetupSkill(a.policy, a.model, kp_scale=a.kp_scale, log=lambda s: print(f"[t={t_sim:6.2f}] {s}"))
    if a.balance:
        mod, cls = a.balance.split(":")
        balance = getattr(importlib.import_module(mod), cls)()
    else:
        balance = StandInBalance()

    r0 = np.loadtxt(a.start_csv, delimiter=",", max_rows=1)
    d.qpos[0:3] = r0[0:3] + np.array([0, 0, 0.01])
    d.qpos[3:7] = r0[[6, 3, 4, 5]]
    d.qpos[qadr] = r0[7:]
    mujoco.mj_forward(m, d)
    sub = int(round(POLICY_DT / m.opt.timestep))

    def state():
        n = a.noise
        return dict(q=d.qpos[qadr] + rng.uniform(-0.01, 0.01, 29) * n, dq=d.qvel[dadr] + rng.uniform(-0.5, 0.5, 29) * n,
                    quat=d.qpos[3:7].copy(), gyro=d.qvel[3:6] + rng.uniform(-0.2, 0.2, 3) * n)

    viewer = importlib.import_module("mujoco.viewer").launch_passive(m, d) if a.viewer else None
    wall0 = time.time()
    mode, t_handoff, result = "getup", None, None
    jump = 0.0
    log = []
    rec = open(a.record, "w") if a.record else None
    if rec:
        rec.write("t,state," + ",".join(f"sim_{n}" for n in SDK_JOINTS) + "," + ",".join(f"des_{n}" for n in SDK_JOINTS)
                  + "," + ",".join(f"ref_{n}" for n in SDK_JOINTS) + ",pelvis_z,tilt\n")
    next_diag = 0.0
    short = [n.replace("_joint", "").replace("left_", "L_").replace("right_", "R_") for n in SDK_JOINTS]
    while True:
        s = state()
        if t_sim >= 1.0 and skill.state == skill.IDLE:
            print(f"[t={t_sim:6.2f}]", skill.start(s)[1])
        if mode == "getup":
            cmd = skill.step(s)
            if skill.stable:
                info = skill.handoff_info()
                balance.start(s, info)
                mode, t_handoff = "balance", t_sim
                print(f"[t={t_sim:6.2f}] 인계 → {'팀원 제어' if a.balance else '임시 대역'} "
                      f"(목표-측정 차이 최대 {np.abs(info['q_des'] - info['q_measured']).max():.3f} rad)")
            elif skill.state == skill.FAILED:
                result = f"FAIL — {skill.reason}"
                break
        else:
            cmd = balance.step(s)
        for _ in range(sub):
            q, dq = d.qpos[qadr], d.qvel[dadr]
            d.ctrl[:] = 0
            d.qfrc_applied[:] = 0
            d.qfrc_applied[dadr] = np.clip(cmd["kp"] * (cmd["q"] - q) + cmd["kd"] * (cmd["dq"] - dq), -effort, effort)
            mujoco.mj_step(m, d)
        t_sim += POLICY_DT
        if skill.state in (skill.GETUP, skill.STANDING) and mode == "getup":
            p = skill.policy
            q_ref = p.to_sdk(p.reference(min(skill.runner.t, p.num_frames - 1))["joint_pos"])
            q_sim = d.qpos[qadr]
            if rec:
                rec.write(f"{t_sim:.3f},{skill.state}," + ",".join(f"{v:.4f}" for v in np.concatenate(
                    [q_sim, cmd["q"], q_ref])) + f",{d.xpos[pelvis][2]:.4f},"
                    f"{np.degrees(np.arccos(np.clip(d.xmat[pelvis][8], -1, 1))):.1f}\n")
            if a.diag and t_sim >= next_diag:
                next_diag = t_sim + 0.5
                err = q_sim - q_ref
                top = np.argsort(-np.abs(err))[:3]
                hip = [SDK_JOINTS.index(n) for n in ("left_hip_roll_joint", "left_hip_yaw_joint",
                                                    "right_hip_roll_joint", "right_hip_yaw_joint")]
                print(f"[diag t={t_sim:5.1f} 기준 {skill.runner.t * POLICY_DT:5.1f}s] "
                      + "  ".join(f"{short[j]} sim{q_sim[j]:+.2f}/명령{cmd['q'][j]:+.2f}/기준{q_ref[j]:+.2f}" for j in top)
                      + " | hip roll/yaw L " + f"{q_sim[hip[0]]:+.2f}/{q_sim[hip[1]]:+.2f}"
                      + " R " + f"{q_sim[hip[2]]:+.2f}/{q_sim[hip[3]]:+.2f}")
        z = d.xpos[pelvis][2]
        tilt = np.degrees(np.arccos(np.clip(d.xmat[pelvis][8], -1, 1)))
        log.append((t_sim, mode, z, tilt))
        if t_handoff is not None and t_sim - t_handoff <= 1.0:
            jump = max(jump, np.abs(d.qvel[3:6]).max())
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            while time.time() - wall0 < t_sim:
                time.sleep(0.001)
        if t_handoff is not None and t_sim - t_handoff >= a.after:
            break
        if t_sim > 60:
            result = "FAIL — 60초 안에 안정 판정 안 됨"
            break

    if result is None:
        z_after = min(z for t, md, z, _ in log if md == "balance")
        result = ("PASS" if z_after > 0.6 else "FAIL") + \
            f" — 시작 후 {t_handoff - 1.0:.1f}s 만에 인계, 인계 후 1초간 몸통 각속도 최대 {jump:.2f} rad/s, " \
            f"인계 후 pelvis 최저 {z_after:.2f} m"
    if rec:
        rec.close()
        print(f"[sim] 기록 저장: {a.record}")
    print("\n판정:", result)
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
