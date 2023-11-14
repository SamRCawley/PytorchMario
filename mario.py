import torch
from torchvision import transforms as T
import numpy as np
from pathlib import Path
import datetime
from MetricLogger import MetricLogger
from wrappers import SkipFrame
# Gym is an OpenAI toolkit for RL
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, TransformObservation
from gym.spaces import Box
# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from agent import Mario

LOGGING = False
VISUALIZE = False

render_mode = 'human' if VISUALIZE else None

env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode=render_mode, apply_api_compatibility=True)

env = JoypadSpace(env, [["right"], ["right", "A"], ["A"], ["left"], ["left", "A"]])
env = SkipFrame(env, skip=4)
#env.observation_space = Box(0, 255, (240, 256, 3), np.uint8)
# Apply Wrappers to environment
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = TransformObservation(env, f=lambda x: np.squeeze(x))
env = FrameStack(env, num_stack=4)


env.reset()
logger_save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
logger_save_dir.mkdir(parents=True)
save_dir = Path('checkpoints')
checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
load_path = save_dir / 'mario_net.chkpt'
if load_path.exists():
    mario.load(load_path)
logger = MetricLogger(logger_save_dir)

episodes = 100

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # 3. Show environment (the visual) [WIP]
        # env.render()

        # 4. Run agent on the state
        action = mario.act(state)
        # 5. Agent performs action
        next_state, reward, done, truncated, info = env.step(action)

        # 6. Remember
        mario.cache(state, next_state, action, reward, done)

        # 7. Learn
        q, loss = mario.learn()

        # 8. Logging
        if LOGGING:
            logger.log_step(reward, loss, q)

        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if done or info['flag_get']:
            break
    if e % 10 == 0:
        print(f'Episode {e}: Current exploration rate {mario.exploration_rate}')

    if LOGGING:
        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e,
                epsilon=mario.exploration_rate,
                step=mario.curr_step
            )
