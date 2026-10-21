import sys
p = sys.argv[1] if len(sys.argv) > 1 else "scripts/rsl_rl/play.py"
s = open(p).read()
if "--from_start" in s:
    print("already patched"); sys.exit()
s = s.replace('''parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")''',
'''parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--from_start", action="store_true", help="Play the motion from frame 0 without randomization.")''', 1)
s = s.replace('''    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs''',
'''    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.from_start:
        env_cfg.episode_length_s = 30.0
        env_cfg.commands.motion.pose_range = {}
        env_cfg.commands.motion.velocity_range = {}
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None''', 1)
s = s.replace('''    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)''',
'''    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args_cli.from_start:
        motion_cmd = env.unwrapped.command_manager.get_term("motion")

        def _start_from_zero(env_ids):
            motion_cmd.time_steps[env_ids] = 0

        motion_cmd._adaptive_sampling = _start_from_zero
        env.reset()''', 1)
assert s.count("from_start") >= 3
open(p, "w").write(s)
print("patched", p)
