class Agent:
    batch_size = 24 # Number of samples to use for Q Learning
    exploration_rate = 0.9 # Percent of time to use random choice (decimal)
    exploration_rate_decay_episodes = 50000 # Percent to reduce exploration by each step (decimal)
    exploration_rate_min = 0.1 # Minimum for exploration rate (decimal). Note: Zero can result in stuck agent
    gamma = 0.9 # Gamma value for Q Learning (weight of future reward)
    burnin = 5e3 # min. experiences before training
    learn_every = 10   # no. of experiences between updates to Q_online
    sync_every = 1e4   # no. of experiences between Q_target & Q_online sync
    save_every = 2e4   # no. of experiences between saving Mario Net
    temp_dir = 'E:/AI_tmp' # location for temp files for replay buffer
    # Probability distibution to use for random action
    # Must match actions in environment
    action_probability = [0.15, 0.15, 0.1, 0.05, 0.05, 0.2, 0.2, 0.05, 0.05]
    learning_rate = 0.00025 # Adam optimizer learning rate

class environment:
    canny_low = 30 # Low threshold for canny edge detection
    canny_high = 180 # Upper threshold for canny edge detection
    skip_frame_num = 8 # Number of frames to skip for each cycle
    stack_frame_num = 2 # Number of frames to stack as series data for model
    save_dir = 'checkpoints' # Directory for save file and log directories
    save_file = 'mario_net.chkpt' #Filename for saved checkpoint
    MARIO_VERSION_OPTIONS = ['SuperMarioBros-v0', 'SuperMarioBrosRandomStages-v0'] #[Stage 1-1, random stage]
    mario_version = MARIO_VERSION_OPTIONS[0] #Pick from MARIO_VERSION_OPTIONS for stage selection
    # Valid button press combinations - list with each element being a list of buttons to press simultaneously as an action
    # e.g. [["right"], ["right", "A"], ["A"], ["left"], ["left", "A"], ["B", "right"], ["B", "right", "A"]]
    actions = [["right"], ["right", "A"], ["A"], ["left"], ["left", "A"], ["B", "right"], ["B", "right", "A"], ["down"], ["up"]]
    num_episodes = 50000 # Number of episodes to train this run (total training = prior training + num_episodes)

class network:
    layer_size = 50 # Number of cells in each hidden layer
    n_layers = 4 # Number of hidden layers
