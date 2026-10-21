#!/usr/bin/env python3
"""
G1 실기 기상 실행기 (PC2, tmux 안에서)

  python3 real_getup.py --policy getup.onnx --model ~/unitree_mujoco/unitree_robots/g1/scene_29dof.xml --net eth0
  python3 real_getup.py ... --balance teammate_balance:Controller     # 안정 판정 뒤 팀원 균형 제어로 인계

조작   리모컨 R1+A 또는 s : 기상 시작 (누운 자세일 때만)     리모컨 B 또는 d : 즉시 damping     q : 종료
--balance 없이 실행하면 인계 대신 기상 정책이 계속 서 있게 유지한다 (단독 시험용).

안전   내장 제어 모드 켜져 있으면 명령 없이 종료(--release 는 누워 있을 때만) / lowstate·제어루프 0.1초 끊김 → damping
       기상 중 기준 대비 60°, 선 뒤 35° 이상 기울면 damping / SIGTERM·SIGHUP·Ctrl+C → damping 2초 후 종료
"""
import argparse
import importlib
import signal
import struct
import sys
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from getup_skill import POLICY_DT, WRIST_SMALL, GetupSkill, tilt_deg

LOWCMD_DT = 0.002
STATE_TIMEOUT = 0.1
MODE_PR = 0
BTN = dict(R1=0, L1=1, start=2, select=3, R2=4, L2=5, F1=6, F2=7, A=8, B=9, X=10, Y=11)


# ====================================================================== 저수준 통신
class G1LowLevel:
    def __init__(self, net: str, release: bool):
        ChannelFactoryInitialize(0, net)
        self.lock = threading.Lock()
        self.msg, self.stamp = None, 0.0
        self.cmd, self.cmd_stamp = None, 0.0
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)
        self.wait_state()

        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        _, result = msc.CheckMode()
        if result.get("name"):
            tilt = tilt_deg(self.msg.imu_state.quaternion)
            if not release:
                sys.exit(f"[real] 내장 제어 모드 '{result['name']}' 켜짐 → 아무 명령도 보내지 않고 종료. "
                         "로봇을 눕힌 뒤 --release, 또는 리모컨 debug 모드(L2+R2).")
            if tilt < 60.0:
                sys.exit(f"[real] 로봇이 누워 있지 않음(pelvis 기울기 {tilt:.0f}°) → 내장 제어 해제 안 하고 종료.")
            while result.get("name"):
                print(f"[real] 내장 모드 '{result['name']}' 해제 중 (누운 상태 확인됨)")
                msc.ReleaseMode()
                time.sleep(1.0)
                _, result = msc.CheckMode()

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.running = True

    def _on_state(self, msg):
        with self.lock:
            self.msg, self.stamp = msg, time.monotonic()

    def wait_state(self, timeout=5.0):
        t0 = time.monotonic()
        while self.msg is None:
            if time.monotonic() - t0 > timeout:
                sys.exit("[real] lowstate 수신 실패: --net 인터페이스 / 로봇 연결 확인")
            time.sleep(0.01)

    def robot(self):
        with self.lock:
            m, age = self.msg, time.monotonic() - self.stamp
        try:
            keys = struct.unpack("H", bytes(m.wireless_remote[2:4]))[0]
        except Exception:
            keys = 0
        return dict(q=np.array([m.motor_state[i].q for i in range(29)]),
                    dq=np.array([m.motor_state[i].dq for i in range(29)]),
                    quat=np.array(m.imu_state.quaternion, float), gyro=np.array(m.imu_state.gyroscope, float),
                    age=age, mode_machine=m.mode_machine,
                    buttons={k: bool(keys >> b & 1) for k, b in BTN.items()})

    def set_cmd(self, cmd):
        with self.lock:
            self.cmd, self.cmd_stamp = cmd, time.monotonic()

    def writer(self):
        nxt = time.perf_counter()
        while self.running:
            r = self.robot()
            with self.lock:
                cmd, cmd_age = self.cmd, time.monotonic() - self.cmd_stamp
            if cmd is None or r["age"] > STATE_TIMEOUT or cmd_age > STATE_TIMEOUT:
                kd = np.full(29, 3.0)
                kd[WRIST_SMALL] = 0.5
                cmd = dict(q=r["q"], dq=np.zeros(29), kp=np.zeros(29), kd=kd)
            lc = self.low_cmd
            lc.mode_pr = MODE_PR
            lc.mode_machine = r["mode_machine"]
            for i in range(29):
                mc = lc.motor_cmd[i]
                mc.mode, mc.tau = 1, 0.0
                mc.q, mc.dq = float(cmd["q"][i]), float(cmd["dq"][i])
                mc.kp, mc.kd = float(cmd["kp"][i]), float(cmd["kd"][i])
            lc.crc = self.crc.Crc(lc)
            self.pub.Write(lc)
            nxt += LOWCMD_DT
            time.sleep(max(0.0, nxt - time.perf_counter()))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--net", default="eth0")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--balance", default=None, help="팀원 균형 제어 'module:Class' (start(state, info), step(state))")
    ap.add_argument("--log", default="real_getup_log.csv")
    a = ap.parse_args()

    skill = GetupSkill(a.policy, a.model, kp_scale=min(a.kp_scale, 1.0))
    balance = None
    if a.balance:
        mod, cls = a.balance.split(":")
        balance = getattr(importlib.import_module(mod), cls)()
    print(f"[real] 기상 {skill.policy.num_frames * POLICY_DT:.1f}s, 선 자세 도달 {skill.standing_frame * POLICY_DT:.1f}s, "
          + ("안정 판정 후 팀원 제어로 인계" if balance else "인계 없음(기상 정책이 계속 유지)"))

    robot = G1LowLevel(a.net, a.release)
    threading.Thread(target=robot.writer, daemon=True).start()
    lock = threading.Lock()
    mode = {"m": "getup"}            # getup → balance

    def do(cmd):
        with lock:
            s = robot.robot()
            if cmd == "start":
                if mode["m"] != "getup":
                    return False, "이미 인계됨"
                return skill.start(s)
            if cmd == "estop":
                skill.abort("ESTOP")
                mode["m"] = "getup"                        # 인계 뒤라도 ESTOP은 damping으로
                return True, "damping"

    logf = open(a.log, "w")
    logf.write("t,mode,state," + ",".join(f"q{i}" for i in range(29)) + ",qw,qx,qy,qz\n")
    t0 = time.monotonic()

    def control_loop():
        nxt, last_print, prev = time.perf_counter(), 0.0, {}
        while robot.running:
            s = robot.robot()
            b = s["buttons"]
            if b["B"]:
                do("estop")
            elif b["R1"] and b["A"] and not (prev.get("R1") and prev.get("A")):
                print("[real] 리모컨 기상 →", do("start")[1])
            prev = b
            with lock:
                try:
                    if s["age"] > STATE_TIMEOUT:
                        skill.abort("lowstate 끊김")
                        mode["m"] = "getup"
                    if mode["m"] == "getup":
                        cmd = skill.step(s)
                        if balance is not None and skill.stable:
                            balance.start(s, skill.handoff_info())
                            mode["m"] = "balance"
                            print("[real] 인계 → 팀원 균형 제어")
                    else:
                        cmd = balance.step(s)
                except Exception as e:
                    skill.abort(f"제어 루프 오류: {e!r}")
                    mode["m"] = "getup"
                    cmd = skill.step(s)
                st = skill.state
            robot.set_cmd(cmd)
            t = time.monotonic() - t0
            logf.write(f"{t:.3f},{mode['m']},{st}," + ",".join(f"{v:.4f}" for v in s["q"]) + ","
                       + ",".join(f"{v:.4f}" for v in s["quat"]) + "\n")
            if t - last_print > 1.0:
                print(f"[real] t={t:6.1f}s {mode['m']:7s} {st:8s} 기울기 {tilt_deg(s['quat']):4.0f}°"
                      + ("  ← 안정, 인계 가능" if skill.stable and mode["m"] == "getup" else "")
                      + (f"  ({skill.reason})" if st == "FAILED" else ""))
                last_print = t
            nxt += POLICY_DT
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def _on_signal(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGHUP, _on_signal)
    threading.Thread(target=control_loop, daemon=True).start()
    print("[real] damping 상태. 리모컨 R1+A 또는 s: 기상 / B 또는 d: damping / q: 종료")
    try:
        while True:
            line = sys.stdin.readline()
            if line == "":
                break
            k = line.strip().lower()
            if k == "s":
                print("[real]", do("start")[1])
            elif k == "d":
                do("estop")
            elif k == "q":
                break
    except KeyboardInterrupt:
        pass
    finally:
        do("estop")
        time.sleep(2.0)
        robot.running = False
        logf.close()
        print("[real] 종료 (damping 상태)")


if __name__ == "__main__":
    main()
