#!/usr/bin/env python3
"""
whole_body_tracking에 기상 전용 task 'Tracking-Getup-G1-v0'를 추가한다.

  cd ~/dev/whole_body_tracking && python3 add_getup_task.py

기존 Tracking-Flat-G1-Wo-State-Estimation-v0 와 관측·행동·보상이 같아서 지금 체크포인트에서 이어서 학습할 수 있다.
바뀌는 것은 '실패를 언제 실패로 보느냐' 뿐이다.

문제: 기존 종료 조건은 높이(z)만 본다 → 인어공주처럼 엉덩이를 바닥에 댄 채 다리를 옆으로 접어도
      pelvis 높이·발 높이·몸통 기울기 크기가 기준과 비슷해서 실패로 안 잡힘.
      그 상태로 몇 초를 버티다 기준이 일어서는 시점(약 10초)에야 실패가 잡히니,
      적응형 샘플링이 엉뚱한 구간(10초)만 집중 연습했다. (학습 로그 sampling_top1_bin ≈ 0.53)
수정:
  1) hip_joint_pos: 고관절 roll·yaw 각도가 기준에서 0.8 rad 이상 벗어나면 종료 → 다리가 옆으로 돌아가는 순간 실패
                   (위치 기준만으로는 인어공주 자세가 안 잡혔음: 확인 재생에서 10.8초에야 ee_body_pos로 종료)
     leg_body_pos : 무릎·발목의 3D 위치(몸통 기준)가 기준에서 0.3 m 이상 벗어나면 종료 (보조)
  2) anchor_ori   : 몸통 기울기 '크기'만이 아니라 '방향'까지 비교 (뒤로 젖혀진 것 / 앞으로 숙인 것 구분)
  3) adaptive_kernel_size 1 → 3 : 실패가 잡힌 구간의 '1~2초 앞'에서도 에피소드를 시작하게 함.
     크기 1이면 실패 구간(5.6초)에서만 시작하는데, 그때는 기준 동작의 속도(관성)를 그대로 받고 시작하니 쉽게 통과한다.
     처음부터 재생하면 그 관성을 스스로 만들어야 해서 실패 → 학습 지표는 좋아 보이는데 실제로는 못 넘어가는 원인.
  4) undesired_contacts 보상 제거 : 기상은 등·엉덩이·팔꿈치가 바닥에 닿아야 하는 동작이라 벌점이 방해만 됨
"""
import os
import re

ROOT = "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1"
CFG = os.path.join(ROOT, "getup_env_cfg.py")
INIT = os.path.join(ROOT, "__init__.py")

CFG_TEXT = '''"""기상 전용 task: 실패 감지를 강화한 Wo-State-Estimation 변형 (add_getup_task.py가 생성)."""
import os

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as isaac_mdp

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatWoStateEstimationEnvCfg


_quat_inv_apply = getattr(math_utils, "quat_apply_inverse", None) or math_utils.quat_rotate_inverse


def bad_joint_pos(env, command_name: str, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """지정 관절의 각도가 기준 동작에서 threshold[rad] 이상 벗어나면 True."""
    command = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    err = torch.abs(command.joint_pos[:, ids] - asset.data.joint_pos[:, ids])
    return torch.any(err > threshold, dim=-1)


def bad_anchor_gravity(env, asset_cfg: SceneEntityCfg, command_name: str, threshold: float) -> torch.Tensor:
    """몸통 좌표계에서 본 중력 방향 벡터 전체를 비교 (기존 anchor_ori는 z 성분만 비교)."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    g_ref = _quat_inv_apply(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    g_rob = _quat_inv_apply(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    return torch.norm(g_ref - g_rob, dim=-1) > threshold


LEG_BODIES = ["left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link"]


@configclass
class G1GetupEnvCfg(G1FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.terminations.anchor_ori = DoneTerm(
            func=bad_anchor_gravity,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.5},
        )
        self.terminations.leg_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos,
            params={"command_name": "motion", "threshold": 0.3, "body_names": LEG_BODIES},
        )
        self.terminations.hip_joint_pos = DoneTerm(
            func=bad_joint_pos,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"]),
                "threshold": 0.8,
            },
        )
        self.commands.motion.adaptive_kernel_size = 3
        self.rewards.undesired_contacts = None


@configclass
class G1GetupRobustEnvCfg(G1GetupEnvCfg):
    """G1GetupEnvCfg + 질량(0.95~1.15배: unitree_mujoco 모델·실기가 학습 URDF보다 무거움)·PD gain 무작위화 (다른 시뮬레이터·실기 차이에 버티도록)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.robot_mass = EventTerm(
            func=isaac_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "mass_distribution_params": (0.95, 1.15), "operation": "scale"},
        )
        # unitree_mujoco 모델은 몸통이 1.77 kg 더 무거움(9.89 vs 8.12 kg) → 몸통에 0~2.5 kg 추가
        self.events.torso_mass = EventTerm(
            func=isaac_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                    "mass_distribution_params": (0.0, 2.5), "operation": "add"},
        )
        self.events.actuator_gains = EventTerm(
            func=isaac_mdp.randomize_actuator_gains,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "stiffness_distribution_params": (0.85, 1.15),
                    "damping_distribution_params": (0.85, 1.15), "operation": "scale"},
        )


@configclass
class G1GetupRobustFakeHandEnvCfg(G1GetupRobustEnvCfg):
    """Robust + 실제 로봇의 가짜 손 (fake_hand.py가 만든 main_fakehand.urdf 사용)."""

    def __post_init__(self):
        super().__post_init__()
        orig = self.scene.robot.spawn.asset_path
        path = os.path.join(os.path.dirname(orig), "main_fakehand.urdf")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 없음 — fake_hand.py를 먼저 실행하세요")
        self.scene.robot.spawn.asset_path = path
'''

REGISTER_TPL = '''

gym.register(
    id="{id}",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={{
        "env_cfg_entry_point": getup_env_cfg.{cls},
        "rsl_rl_cfg_entry_point": f"{{agents.__name__}}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    }},
)
'''
TASKS = [("Tracking-Getup-G1-v0", "G1GetupEnvCfg"), ("Tracking-Getup-Robust-G1-v0", "G1GetupRobustEnvCfg"),
         ("Tracking-Getup-Robust-FakeHand-G1-v0", "G1GetupRobustFakeHandEnvCfg")]
EXPORTER = "source/whole_body_tracking/whole_body_tracking/utils/exporter.py"

assert os.path.isdir(ROOT), "whole_body_tracking 루트 폴더에서 실행하세요"
with open(CFG, "w") as f:
    f.write(CFG_TEXT)
print("작성:", CFG)

s = open(INIT).read()
if "getup_env_cfg" not in s:
    s, n = re.subn(r"from \. import agents, flat_env_cfg\b", "from . import agents, flat_env_cfg, getup_env_cfg", s, count=1)
    assert n == 1, "__init__.py 의 import 줄을 찾지 못함"
for tid, cls in TASKS:
    if tid in s:
        print("이미 등록됨:", tid)
    else:
        s = s.rstrip() + "\n" + REGISTER_TPL.format(id=tid, cls=cls)
        print("등록:", tid)
open(INIT, "w").write(s)

# PD gain을 무작위화해도 ONNX 메타데이터에는 '명목값'이 들어가도록
e = open(EXPORTER).read()
e2 = e.replace('data.joint_stiffness[0]', 'data.default_joint_stiffness[0]').replace('data.joint_damping[0]', 'data.default_joint_damping[0]')
if e2 != e:
    open(EXPORTER, "w").write(e2)
    print("exporter: 메타데이터 gain을 default_joint_stiffness/damping으로 변경")
