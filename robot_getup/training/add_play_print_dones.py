#!/usr/bin/env python3
"""play.py --from_start 실행 중 '모든 env'의 에피소드 종료 시각과 원인을 출력 (여러 env로 성공률 확인용).
   cd ~/dev/whole_body_tracking && python3 add_play_print_dones.py scripts/rsl_rl/play.py
   (이전 버전 훅이 있으면 새 버전으로 교체)"""
import re, sys
p = sys.argv[1] if len(sys.argv) > 1 else "scripts/rsl_rl/play.py"
s = open(p).read()
s = re.sub(r"            if args_cli\.from_start:\n                _ep_steps = .*?globals\(\)\[\"_ep_steps\"\] = 0\n", "", s, flags=re.S)
anchor = "            obs, _, dones, _ = env.step(actions)\n"
assert anchor in s, "env.step 줄을 찾지 못함"
hook = '''            if args_cli.from_start:
                _ep = globals().setdefault("_ep_steps", torch.zeros(len(dones), dtype=torch.long, device=dones.device))
                _ep += 1
                _done_ids = torch.nonzero(dones).flatten().tolist()
                if _done_ids:
                    _tm = env.unwrapped.termination_manager
                    for _i in _done_ids:
                        _hits = [n for n in _tm.active_terms if bool(_tm.get_term(n)[_i])]
                        print(f"[from_start] env {_i:3d} 종료 t={int(_ep[_i]) * env.unwrapped.step_dt:5.2f}s 원인={_hits}")
                    _ep[_done_ids] = 0
'''
s = s.replace(anchor, anchor + hook, 1)
if "\nimport torch" not in s:
    s = s.replace("import gymnasium as gym\n", "import gymnasium as gym\nimport torch\n", 1)
open(p, "w").write(s)
print("patched", p)
