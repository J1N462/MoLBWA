# G1 기상 스킬 — 누움 → 서기 → 균형 제어(팀원)에게 인계

```
IDLE(누움, damping) ─start─▶ ALIGN(2초) ─▶ GETUP(정책) ─▶ STANDING(정책이 계속 붙잡고 안정 판정) ─stable─▶ 팀원 균형 제어
         ▲                                                                                     │
         └──────────────────── abort / ESTOP / 안전 검사 → damping ◀──────────────────────────┘
```
우리 담당은 **`stable == True`가 되는 순간까지**입니다. 그 뒤는 팀원 균형 제어가 이어받습니다.

## 파일

| 파일 | 역할 |
|---|---|
| `getup_skill.py` | 기상 스킬 본체. SDK·시뮬레이터 의존 없음 → **팀원 코드에 그대로 import** |
| `sim_getup.py` | MuJoCo: 기상 → 안정 판정 → 인계(임시 대역 또는 팀원 클래스)까지 시험 |
| `real_getup.py` | 실기: 단독 기상 시험, 또는 `--balance`로 팀원 제어까지 이어서 실행 |
| `stand_pose.json` | 기상 끝 선 자세(참고용). 실제 인계 자세는 매번 `handoff_info()`로 받음 |

필요한 파일: `getup.onnx`(기상 동작이 안에 들어 있음, sim·실기 같은 파일), MuJoCo `scene_29dof.xml`(실기에선 몸통 자세 FK 계산용).
`g1_getup.csv`는 시뮬에서 로봇을 눕혀 놓는 데만 씁니다.

---

## 인계 규약 (팀원과 맞출 것)

**호출 주기** 50 Hz (0.02 s). lowcmd는 그대로 500 Hz로 계속 보내면 됨 (마지막 명령 유지).

**입력 state** (SDK 모터 순서, rad / rad/s)
```python
{"q": [29], "dq": [29], "quat": pelvis IMU [w, x, y, z], "gyro": pelvis 각속도 [3]}   # lowstate 그대로
```
**출력 cmd** (SDK 모터 순서 29개) → `motor_cmd[i].q/dq/kp/kd`, tau=0
```python
{"q": [29], "dq": [29], "kp": [29], "kd": [29]}
```

**`stable`이 True가 되는 조건** (STANDING 진입 후 모두 0.5초 연속)
- pelvis 기울기 < 10°, 몸통 각속도 < 0.5 rad/s, 모든 관절 속도 < 1.0 rad/s
- 기상 시작(ALIGN 포함)부터 약 15.5초 뒤 (원본 로그 기준 선 자세 도달 13.0초 + ALIGN 2초 + 0.5초)

**`handoff_info()`** — 인계 순간 값
| 키 | 내용 |
|---|---|
| `q_measured` | 인계 순간 측정 관절각 → **팀원 제어의 시작 자세로 쓰기** |
| `q_des`, `kp`, `kd` | 그 순간 우리가 보내던 목표·gain → 팀원 gain으로 0.2~0.5초 블렌딩하면 충격 없음 |
| `stand_pose` | reference 마지막 선 자세 |
| `tilt_deg` | 인계 순간 기울기 |

우리 gain은 낮은 편입니다 (학습 조건: hip pitch/yaw ≈ 40, hip roll·knee ≈ 99, ankle·waist roll/pitch ≈ 28.5, 팔 ≈ 14).
팀원 제어 gain이 크게 다르면 **바로 바꾸지 말고 블렌딩**하세요 (`sim_getup.py`의 `StandInBalance` 참고).

**제어권**: `rt/lowcmd`를 보내는 프로그램은 하나만. 아래 둘 중 하나로 합칩니다.

(A) 팀원 메인 루프에 import (권장)
```python
from getup_skill import GetupSkill
skill = GetupSkill("getup.onnx", "scene_29dof.xml")
skill.start(state)                       # 세우기 버튼
mode = "getup"
while True:                              # 50 Hz
    if mode == "getup":
        cmd = skill.step(state)
        if skill.stable:
            balance.start(state, skill.handoff_info()); mode = "balance"
    else:
        cmd = balance.step(state)
    send_lowcmd(cmd)
```
(B) 우리 실행기에 팀원 클래스를 꽂기
```python
class Controller:                        # teammate_balance.py
    def start(self, state, handoff_info): ...
    def step(self, state): return {"q":..., "dq":..., "kp":..., "kd":...}
```
```bash
python3 real_getup.py --policy getup.onnx --model <scene_29dof.xml> --balance teammate_balance:Controller
python3 sim_getup.py  --policy getup.onnx --start-csv g1_getup.csv --model <scene_29dof.xml> --balance teammate_balance:Controller
```
(B)는 팀원 클래스를 **sim에서 먼저** 인계 시험할 수 있다는 장점이 있습니다.

---

## 남은 과정

1. **학습 끝 → ONNX 내보내기** (학습 PC)
   ```bash
   python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-Wo-State-Estimation-v0 --num_envs=2 \
       --headless --enable_cameras --video --video_length 1000 \
       --wandb_path=incheon-national-university/g1_getup/<run id>
   cp logs/rsl_rl/temp/exported/policy.onnx ~/g1_data/getup.onnx
   ```
2. **sim2sim** (개발 PC) — 모두 PASS
   ```bash
   S="--policy ~/g1_data/getup.onnx --start-csv ~/g1_data/g1_getup.csv --model ~/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
   python3 sim_getup.py $S --viewer
   python3 sim_getup.py $S --friction 0.6
   python3 sim_getup.py $S --kp-scale 0.9 --seed 1
   python3 sim_getup.py $S --noise 2 --seed 2
   ```
   볼 것: 안정 판정까지 걸린 시간, 인계 후 1초간 몸통 각속도(작을수록 부드러운 인계)
3. **PC2 준비** — cyclonedds 0.10.x + unitree_sdk2_python, `pip3 install mujoco onnxruntime numpy`, unitree_mujoco 클론
4. **실기 단독 기상** (갠트리, tmux)
   ```bash
   python3 real_getup.py --policy ~/getup.onnx --model ~/unitree_mujoco/unitree_robots/g1/scene_29dof.xml --net eth0
   ```
   B로 damping 전환 먼저 확인 → `s` → `← 안정, 인계 가능`이 뜨는지 → 줄을 조금씩 풀며 반복
5. **팀원 제어와 인계 시험**: sim에서 `--balance`로 먼저 → 실기에서 같은 클래스로
