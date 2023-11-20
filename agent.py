import torch
import numpy as np
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, LazyTensorStorage
from neuralnet import MarioNet
from tensordict import TensorDict

#torch.autograd.set_detect_anomaly(True)

class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        self.storage_device = self.device
        print('Using: ', self.device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        #self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, scratch_dir=Path('tmp'), device=self.device))
        #Memory settings need to be configured based on the host machine
        self.memory = TensorDictReplayBuffer(storage=LazyTensorStorage(5000, device=self.storage_device))
        self.batch_size = 32
        self.net_tuple = None
        self.prev_net_tuple = None
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9
        self.curr_step = 0
        self.burnin = 1e4 # min. experiences before training
        self.learn_every = 3   # no. of experiences between updates to Q_online
        self.sync_every = 1e4   # no. of experiences between Q_target & Q_online sync
        self.save_every = 2e4   # no. of experiences between saving Mario Net
        self.save_dir = save_dir
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device=self.device)
        if checkpoint:
            self.load(checkpoint)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def reset(self):
        self.net_tuple = None
        self.prev_net_tuple = None

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
            #[["right"], ["right", "A"], ["A"], ["left"], ["left", "A"], ['B', 'right'], ['B, right', 'A']]
            action_idx = np.random.choice(np.arange(self.action_dim), p=[0.2, 0.2, 0.1, 0.05, 0.05, 0.2, 0.2])
        # EXPLOIT
        else:
            state = torch.tensor(state, device=self.device)
            state = state.unsqueeze(1)
            self.prev_net_tuple = self.net_tuple
            action_values, self.net_tuple = self.net(state, self.net_tuple, model='online')
            action_idx = torch.argmax(action_values).item()
        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        # increment step
        self.curr_step += 1
        if action_idx > self.action_dim:
            raise ValueError(f'Action index of {action_idx} > action dimension {self.action_dim}')
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
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)
        #Add state and h_t, c_t tuple to memory for learning
        if self.prev_net_tuple is not None and self.net_tuple is not None:
            self.memory.add(TensorDict({"state": state, "h_n": self.prev_net_tuple[0].squeeze(),  "c_n": self.prev_net_tuple[1].squeeze(), "next_state": next_state, "next_h_n": self.net_tuple[0].squeeze(), "next_c_n": self.net_tuple[1].squeeze(), "action": action, "reward": reward, "done": done}, batch_size=[], device=self.storage_device))


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, h_n, c_n, next_state, next_h_n, next_c_n, action, reward, done = (batch.get(key) for key in ("state", "h_n", "c_n", "next_state", "next_h_n", "next_c_n", "action", "reward", "done"))
        return state.transpose(0,1).contiguous(), (h_n.transpose(0,1).contiguous().detach(), c_n.transpose(0,1).contiguous().detach()), next_state.transpose(0,1).contiguous(), (next_h_n.transpose(0,1).contiguous().detach(), next_c_n.transpose(0,1).contiguous().detach()), action.squeeze(), reward.squeeze(), done.squeeze()


    def td_estimate(self, state, action, hidden):
        net_result, online_tuple = self.net(state, hidden, model='online')
        current_Q = net_result[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done, hidden, next_hidden):
        next_state_Q, online_tuple = self.net(next_state, next_hidden, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q, target_tuple = self.net(next_state, next_hidden, model='target')
        next_Q = next_Q[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target) :
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


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
        state, hidden, next_state, next_hidden, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action, hidden)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done, hidden, next_hidden)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


    def save(self):
        save_path = self.save_dir / f"mario_net.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
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

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate