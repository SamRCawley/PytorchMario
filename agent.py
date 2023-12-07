import torch
import numpy as np
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, LazyTensorStorage
from neuralnet import MarioNet
from tensordict import TensorDict
from pathlib import Path
from config import Agent as config
from config import network as net_config
from config import environment as env_config
#torch.autograd.set_detect_anomaly(True)

class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        self.storage_device = torch.device('cpu')
        print('Using: ', self.device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = config.batch_size
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(20000, scratch_dir=Path(config.temp_dir), device=self.storage_device), pin_memory=True, batch_size=self.batch_size, prefetch=100)
        self.episode_states, self.episode_next_states, self.episode_actions, self.episode_rewards, self.episode_done = [], [], [], [], []
        #Memory settings need to be configured based on the host machine
        #self.memory = TensorDictReplayBuffer(storage=LazyTensorStorage(3000, device=self.storage_device))
        self.episode_num = 0
        self.exploration_rate = config.exploration_rate
        self.exploration_rate_decay = config.exploration_rate_decay
        self.exploration_rate_min = config.exploration_rate_min
        self.gamma = config.gamma
        self.curr_step = 0
        self.burnin = config.burnin
        self.learn_every = config.learn_every
        self.sync_every = config.sync_every
        self.save_every = config.save_every
        self.save_dir = save_dir
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device=self.device)
        if checkpoint:
            self.load(checkpoint)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def reset(self):
        if self.episode_states:
            #Adjust rewards in last few actions before failure to compensate for unwinnable situations (e.g. falling into pits)
            self.episode_rewards = np.array(self.episode_rewards)
            self.episode_rewards[-5:-1] -= 15
            episode = TensorDict({"state": self.episode_states, "next_state": self.episode_next_states, "action": self.episode_actions, "reward": self.episode_rewards, "done": self.episode_done}, batch_size=len(self.episode_states), device=self.storage_device)
            self.memory.extend(episode)
            self.episode_states, self.episode_next_states, self.episode_actions, self.episode_rewards, self.episode_done = [], [], [], [], []
        self.episode_num += 1

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """

        if isinstance(state, tuple):
            state = state[0]
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.choice(np.arange(self.action_dim), p=config.action_probability)
        # EXPLOIT
        else:
            state = torch.as_tensor(state, device=self.device)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values)
            action_idx = action_idx.to('cpu', non_blocking=True).item()
        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """

        if isinstance(state, tuple):
            state = state[0]
        if not isinstance(state, np.ndarray):
            state =np.array(state, dtype=np.float32)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        state = torch.as_tensor(state, device=self.storage_device)
        next_state = torch.as_tensor(next_state, device=self.storage_device)
        action = torch.tensor([action], device=self.storage_device)
        reward = torch.tensor([reward], device=self.storage_device)
        done = torch.tensor([done], device=self.storage_device)
        #self.temp.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[], device=self.storage_device))
        self.episode_states.append(state)
        self.episode_next_states.append(next_state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_done.append(done)

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample().to(self.device, non_blocking=True)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state , action.squeeze(), reward.squeeze(), done.squeeze()


    def td_estimate(self, state, action):
        net_result = self.net(state, model='online')
        current_Q = net_result[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')
        next_Q = next_Q[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target) :
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()


    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.detach().mean().item(), loss)


    def save(self):
        save_path = self.save_dir / f"mario_net.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate,
                episode=self.episode_num
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=self.device)
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')
        episode_num = ckp.get('episode')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
        self.episode_num = episode_num
