import argparse
import torch
import numpy as np
from pathlib import Path
import datetime
from MetricLogger import MetricLogger
from wrappers import SkipFrame
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, TransformObservation, FlattenObservation
from nes_py.wrappers import JoypadSpace
import cv2
import gym_super_mario_bros
from agent import Mario
from config import environment as config
from torch.profiler import profile, record_function, ProfilerActivity
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--play', '-p', action="store_true", help='Runs model from checkpoint with visualization but no learning')
parser.add_argument('--visualize', '-v', action="store_true", help='Enables visualization of the game')
parser.add_argument('--no_log', '-nl', action="store_false", help='Disables automatic logging')
parser.add_argument('--num_episodes', '-n', default=config.num_episodes, type=int, help=f'Sets number of episodes (default={config.num_episodes})')
args = parser.parse_args()
LOGGING = args.no_log
VISUALIZE = args.visualize
NUM_EPISODES = args.num_episodes
if args.play:
    LOGGING = False
    VISUALIZE = True


render_mode = 'human' if VISUALIZE else None

env = gym_super_mario_bros.make(config.mario_version, render_mode=render_mode, apply_api_compatibility=True)

"""#Debug for edge detection filter
def cannyfilter (x):
    edges = cv2.Canny(x, 30, 180)
    cv2.imshow('game', edges)
    cv2.waitKey(0)
    return edges"""

env = JoypadSpace(env, config.actions)
env = SkipFrame(env, skip=config.skip_frame_num)
# Apply Wrappers to environment
#env = GrayScaleObservation(env, keep_dim=False)
#env = TransformObservation(env, f=lambda x: cv2.Canny(x, config.canny_low, config.canny_high))
#env = TransformObservation(env, f=lambda x: cannyfilter(x))
env = TransformObservation(env, f=lambda x: (x / 255.))
#env = TransformObservation(env, f=lambda x: np.squeeze(x))
#env = FlattenObservation(env)
env = TransformObservation(env, f= lambda x: x.astype(np.float32))
env = FrameStack(env, num_stack=config.stack_frame_num)
env = TransformObservation(env, f=lambda x: np.moveaxis(x, [0, 3], [1, 0]) )
state = env.reset()[0]

if LOGGING:
    logger_save_dir = Path(config.save_dir) / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    logger_save_dir.mkdir(parents=True)
    logger = MetricLogger(logger_save_dir)
save_dir = Path(config.save_dir)
load_path = save_dir / config.save_file
checkpoint = None
if load_path.exists():
    checkpoint = load_path
if args.play and not load_path.exists():
    raise Exception("Cannot run replay:  No checkpoint exists.")
mario = Mario(state_dim=state.shape, action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
if args.play:
    mario.exploration_rate = mario.exploration_rate_min
#with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
    ### for Loop that train the model num_episodes times by playing the game
for e in range(NUM_EPISODES):
    state = env.reset()
    mario.reset()
    # Play the game!
    prev_lives = 3
    lives = 3
    while True:

        # 4. Run agent on the state
        action = mario.act(state)
        # 5. Agent performs action
        next_state, reward, done, truncated, info = env.step(action)
        prev_lives = lives
        lives = info['life']
        lost_life = lives < prev_lives
        if not args.play:
            # 6. Remember
            mario.cache(state, next_state, action, reward, lost_life)

            # 7. Learn
            q, loss = mario.learn()

        # 8. Logging
        if LOGGING:
            logger.log_step(reward, loss, q)

        # 9. Update state
        state = next_state

        # 10. Check if end of game, level, or life lost
        if lost_life:
            mario.decrease_recent_rewards()
        if info['flag_get']:
            mario.increase_level_rewards()
        if done:
            break


    if e % 10 == 0:
        print(f'Episode {mario.episode_num}, Step {mario.curr_step}: Current exploration rate {mario.exploration_rate}')

    if LOGGING:
        logger.log_episode()

        if e % 20 == 0 and mario.curr_step > mario.burnin:
            logger.record(
                episode=mario.episode_num,
                epsilon=mario.exploration_rate,
                step=mario.curr_step
            )
#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#print(prof.key_averages(group_by_stack_n=10).table(sort_by="cpu_time_total", row_limit=5))
#print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
#print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_memory_usage", row_limit=5))
#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
