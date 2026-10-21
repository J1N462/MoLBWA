"""
G1 기상 스킬 — 누운 자세에서 일어나 '안정적으로 섰다'고 판단되면 균형 제어(팀원 코드)에게 넘긴다.
SDK·시뮬레이터에 의존하지 않는 순수 모듈: 상태를 넣으면 관절 명령이 나온다 (50 Hz 호출).

    skill = GetupSkill("getup.onnx", "scene_29dof.xml")
    ok, msg = skill.start(state)            # 누운 자세 확인 후 시작
    while ...:                              # 50 Hz
        cmd = skill.step(state)             # {"q","dq","kp","kd"} 29개, SDK 모터 순서
        if skill.stable:                    # 서서 안정 → 인계
            info = skill.handoff_info()     # 인계 순간의 측정 자세 / 마지막 목표 / gain
            ...팀원 균형 제어로 전환...

state = {"q": 29, "dq": 29 (SDK 순서), "quat": pelvis IMU [w,x,y,z], "gyro": pelvis 각속도 [3] (몸체 좌표)}
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import mujoco

SDK_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
POLICY_DT = 0.02
WRIST_SMALL = [20, 21, 27, 28]      # 손목 pitch/yaw (작은 모터)


# ------------------------------------------------------------------ quaternion (w, x, y, z)
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def gravity_in_body(q):
    return quat_to_mat(np.asarray(q, float)).T @ np.array([0.0, 0.0, -1.0])


def tilt_deg(q):
    return float(np.degrees(np.arccos(np.clip(-gravity_in_body(q)[2], -1, 1))))


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return 0.5 - 0.5 * np.cos(np.pi * x)


# ------------------------------------------------------------------ ONNX 정책
def _csv(s, typ=float):
    return [typ(v) for v in s.split(",")] if s else []


class MotionPolicy:
    """BeyondMimic ONNX: inputs (obs[1,N], time_step[1,1]) → actions + 기준 동작(joint_pos, body_quat_w ...).
    기준 동작(누움→서기 등)은 이 ONNX 안에 들어 있다. sim2sim과 실기 모두 같은 ONNX 파일을 쓴다."""

    def __init__(self, path: str):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.path = path
        self.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.joint_names = _csv(meta["joint_names"], str)
        self.kp = np.array(_csv(meta["joint_stiffness"]))
        self.kd = np.array(_csv(meta["joint_damping"]))
        self.default_q = np.array(_csv(meta["default_joint_pos"]))
        self.action_scale = np.array(_csv(meta["action_scale"]))
        self.obs_names = _csv(meta["observation_names"], str)
        self.body_names = _csv(meta["body_names"], str)
        self.anchor_idx = self.body_names.index(meta["anchor_body_name"])
        self.obs_dim = self.sess.get_inputs()[0].shape[1]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        assert len(self.joint_names) == 29, "29dof 정책이 아닙니다"
        self.i2s = np.array([SDK_JOINTS.index(n) for n in self.joint_names])   # isaac i → sdk index
        self.s2i = np.argsort(self.i2s)                                        # sdk j → isaac index
        self._zeros = np.zeros((1, self.obs_dim), np.float32)
        self.num_frames = self._find_num_frames()

    def _run(self, obs, t):
        out = self.sess.run(None, {"obs": obs.reshape(1, -1).astype(np.float32),
                                   "time_step": np.array([[float(t)]], np.float32)})
        return dict(zip(self.out_names, out))

    def reference(self, t):
        o = self._run(self._zeros, t)
        return dict(joint_pos=o["joint_pos"][0].astype(float), joint_vel=o["joint_vel"][0].astype(float),
                    anchor_quat=o["body_quat_w"][0, self.anchor_idx].astype(float))

    def act(self, obs, t):
        return self._run(obs, t)["actions"][0].astype(float)

    def to_sdk(self, v_isaac):
        out = np.zeros(29)
        out[self.i2s] = v_isaac
        return out

    def _find_num_frames(self):
        last = self.reference(10 ** 7)["joint_pos"]
        lo, hi = 0, 10 ** 7
        while lo < hi:
            mid = (lo + hi) // 2
            if np.array_equal(self.reference(mid)["joint_pos"], last):
                hi = mid
            else:
                lo = mid + 1
        return lo + 1


# ------------------------------------------------------------------ torso FK
class TorsoFK:
    """pelvis 자세(IMU) + 관절각 → torso_link 월드 자세. MuJoCo 모델을 FK 계산기로만 사용."""

    def __init__(self, model_path: str):
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in SDK_JOINTS])
        self.torso = self.m.body("torso_link").id

    def torso_quat(self, pelvis_quat, q_sdk):
        self.d.qpos[:] = 0
        self.d.qpos[3:7] = pelvis_quat
        self.d.qpos[self.qadr] = q_sdk
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xquat[self.torso].copy()


# ------------------------------------------------------------------ 정책 1개를 처음부터 끝까지 재생
class MotionRunner:
    def __init__(self, policy: MotionPolicy):
        self.p = policy
        self.t = 0
        self.last_action = np.zeros(29)
        self.world_to_init = np.array([1.0, 0, 0, 0])
        self.ref0 = policy.reference(0)
        self.last_obs = None
        self.ref_anchor = self.ref0["anchor_quat"]

    @property
    def done(self):
        return self.t >= self.p.num_frames - 1

    def begin(self, torso_q):
        """기준 동작 첫 프레임의 yaw를 현재 로봇 yaw에 맞추고 처음부터 시작."""
        self.world_to_init = quat_mul(yaw_quat(torso_q), quat_inv(yaw_quat(self.ref0["anchor_quat"])))
        self.t = 0
        self.last_action = np.zeros(29)

    def peek_anchor(self):
        ref = self.p.reference(min(self.t, self.p.num_frames - 1))
        return quat_mul(self.world_to_init, ref["anchor_quat"])

    def step(self, robot, torso_q, obs_override_sdk=None):
        """obs_override_sdk: 관측에서 '기준값'으로 바꿔 넣을 SDK 관절 목록 (팔을 스킬이 덮어쓸 때 선택 사항)."""
        p = self.p
        t = min(self.t, p.num_frames - 1)
        ref = p.reference(t)
        self.ref_anchor = quat_mul(self.world_to_init, ref["anchor_quat"])
        q_i = robot["q"][p.i2s].copy()
        dq_i = robot["dq"][p.i2s].copy()
        if obs_override_sdk:
            ii = p.s2i[list(obs_override_sdk)]
            q_i[ii] = ref["joint_pos"][ii]
            dq_i[ii] = ref["joint_vel"][ii]
        rel = quat_mul(quat_inv(torso_q), self.ref_anchor)
        terms = {
            "command": np.concatenate([ref["joint_pos"], ref["joint_vel"]]),
            "motion_anchor_ori_b": quat_to_mat(rel)[:, :2].reshape(-1),
            "base_ang_vel": robot["gyro"],
            "joint_pos": q_i - p.default_q,
            "joint_vel": dq_i,
            "actions": self.last_action,
        }
        missing = [n for n in p.obs_names if n not in terms]
        if missing:
            raise RuntimeError(f"구현되지 않은 관측 항목: {missing} (task가 Wo-State-Estimation인지 확인)")
        obs = np.concatenate([terms[n] for n in p.obs_names])
        if obs.shape[0] != p.obs_dim:
            raise RuntimeError(f"관측 크기 불일치: {obs.shape[0]} vs 정책 {p.obs_dim}")
        self.last_obs = obs
        action = p.act(obs, t)
        self.last_action = action
        self.t += 1
        return p.to_sdk(p.default_q + action * p.action_scale)



# ------------------------------------------------------------------ 기상 스킬
class GetupSkill:
    IDLE, ALIGN, GETUP, STANDING, FAILED = "IDLE", "ALIGN", "GETUP", "STANDING", "FAILED"

    def __init__(self, policy_path: str, model_xml: str, align_time=2.0, kp_scale=1.0,
                 start_check_deg=30.0, motion_ori_limit_deg=60.0, stand_tilt_limit_deg=35.0,
                 stable_tilt_deg=10.0, stable_gyro=0.5, stable_dq=1.0, stable_time=0.5, damp_kd=3.0, log=print):
        self.policy = MotionPolicy(policy_path)
        self.fk = TorsoFK(model_xml)
        self.runner = MotionRunner(self.policy)
        self.align_time, self.kp_scale, self.damp_kd = align_time, kp_scale, damp_kd
        self.start_check = np.radians(start_check_deg)
        self.motion_ori_limit = np.radians(motion_ori_limit_deg)
        self.stand_tilt_limit = np.radians(stand_tilt_limit_deg)
        self.stable_tilt, self.stable_gyro, self.stable_dq = np.radians(stable_tilt_deg), stable_gyro, stable_dq
        self.stable_time = stable_time
        self.log = log
        self.q0 = self.policy.to_sdk(self.runner.ref0["joint_pos"])
        self.stand_pose = self.policy.to_sdk(self.policy.reference(10 ** 7)["joint_pos"])
        self.standing_frame = self._find_standing_frame()
        self.state, self.reason = self.IDLE, ""
        self._stable_since = None
        self._last_cmd = None
        self._last_state = None
        self._last_t = 0.0

    # ---------------- 외부 API
    def start(self, s):
        if self.state not in (self.IDLE, self.FAILED):
            return False, f"이미 실행 중 ({self.state})"
        err = self._ori_err(self.fk.torso_quat(s["quat"], s["q"]), self.runner.ref0["anchor_quat"])
        if err > self.start_check:
            return False, f"시작 거부: 누운 기준 자세와 {np.degrees(err):.0f}° 다름 (등을 대고 눕혀 주세요)"
        self._align_from, self._align_k = np.asarray(s["q"], float).copy(), 0
        self._dq_f = self._gy_f = None
        self.state, self.reason, self._stable_since = self.ALIGN, "", None
        self.log(f"[getup] ALIGN {self.align_time:.1f}s")
        return True, "기상 시작"

    def abort(self, reason="abort"):
        if self.state not in (self.IDLE, self.FAILED):
            self.log(f"[getup] 중단 → damping ({reason})")
        self.state, self.reason = self.FAILED, reason

    @property
    def stable(self):
        """서 있고, 기울기·각속도·관절속도가 기준 이하로 stable_time 이상 유지됨 → 인계해도 됨."""
        return self.state == self.STANDING and self._stable_since is not None \
            and self._last_t - self._stable_since >= self.stable_time

    def handoff_info(self):
        s, c = self._last_state, self._last_cmd
        return dict(q_measured=s["q"].copy(), q_des=c["q"].copy(), kp=c["kp"].copy(), kd=c["kd"].copy(),
                    stand_pose=self.stand_pose.copy(), tilt_deg=tilt_deg(s["quat"]))

    def step(self, s):
        s = {k: np.asarray(v, float) for k, v in s.items() if k in ("q", "dq", "quat", "gyro")}
        self._last_state = s
        self._last_t += POLICY_DT
        if self.state in (self.IDLE, self.FAILED):
            cmd = self._damp(s)
        elif self.state == self.ALIGN:
            self._align_k += 1
            w = smoothstep(self._align_k * POLICY_DT / self.align_time)
            cmd = self._cmd((1 - w) * self._align_from + w * self.q0)
            if w >= 1.0:
                self.runner.begin(self.fk.torso_quat(s["quat"], s["q"]))
                self.state = self.GETUP
                self.log(f"[getup] GETUP (기준 {self.policy.num_frames * POLICY_DT:.1f}s, "
                         f"선 자세 도달 예정 {self.standing_frame * POLICY_DT:.1f}s)")
        else:
            torso = self.fk.torso_quat(s["quat"], s["q"])
            if not self._safe(torso, s):
                cmd = self._damp(s)
            else:
                cmd = self._cmd(self.runner.step(s, torso))
                if self.state == self.GETUP and self.runner.t >= self.standing_frame:
                    self.state = self.STANDING
                    self.log("[getup] STANDING: 선 자세 도달, 안정 판정 중")
                if self.state == self.STANDING:
                    self._update_stable(s)
        self._last_cmd = cmd
        return cmd

    # ---------------- 내부
    def _find_standing_frame(self):
        z = np.array([self.policy._run(self.policy._zeros, t)["body_pos_w"][0, 0, 2]
                      for t in range(self.policy.num_frames)])
        return int(np.argmax(z > z[-1] - 0.02))

    def _update_stable(self, s):
        # 센서 잡음에 흔들리지 않도록 0.2초 저역통과 후 판정
        a = POLICY_DT / (0.2 + POLICY_DT)
        self._dq_f = s["dq"].copy() if getattr(self, "_dq_f", None) is None else (1 - a) * self._dq_f + a * s["dq"]
        self._gy_f = s["gyro"].copy() if getattr(self, "_gy_f", None) is None else (1 - a) * self._gy_f + a * s["gyro"]
        ok = (np.radians(tilt_deg(s["quat"])) < self.stable_tilt and np.abs(self._gy_f).max() < self.stable_gyro
              and np.abs(self._dq_f).max() < self.stable_dq)
        if ok and self._stable_since is None:
            self._stable_since = self._last_t
        elif not ok:
            self._stable_since = None
        if self.stable and not getattr(self, "_announced", False):
            self._announced = True
            self.log(f"[getup] 안정 → 인계 가능 (기울기 {tilt_deg(s['quat']):.1f}°)")
        if not self.stable:
            self._announced = False

    @staticmethod
    def _ori_err(q_a, q_b):
        return np.arccos(np.clip(np.dot(gravity_in_body(q_a), gravity_in_body(q_b)), -1, 1))

    def _safe(self, torso, s):
        if self.state == self.GETUP:
            err = self._ori_err(torso, self.runner.peek_anchor())
            if err > self.motion_ori_limit:
                self.abort(f"기상 중 몸통 자세가 기준과 {np.degrees(err):.0f}° 어긋남")
                return False
        elif np.radians(tilt_deg(s["quat"])) > self.stand_tilt_limit:
            self.abort(f"선 뒤 기울기 {tilt_deg(s['quat']):.0f}°")
            return False
        return True

    def _cmd(self, q_des):
        p = self.policy
        return dict(q=np.asarray(q_des, float).copy(), dq=np.zeros(29),
                    kp=p.to_sdk(p.kp) * self.kp_scale, kd=p.to_sdk(p.kd))

    def _damp(self, s):
        kd = np.full(29, self.damp_kd)
        kd[WRIST_SMALL] = 0.5
        return dict(q=s["q"].copy(), dq=np.zeros(29), kp=np.zeros(29), kd=kd)
