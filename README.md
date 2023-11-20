# PytorchMario
## An updated version of the Mario RL tutorial from pytorch.

This is an update of the pytorch tutorial to create a RL model for Super Mario Bros with a few notable changes
* The model has been changed from sequential DDQN to LSTM DDQN.  This should be a better fit since actions in mario affect more than just current state
* It does NOT use simplified block mario (I feel this cheats at the real task of playing mario by RL, although you could argue it's a filter you could create on the image data, but that only holds as long as sprites stay consistent)
* Uses native resolution instead of resize to 84x84 (lack of guarentee of how resize is going to treat small sprites)
* Adds canny edge detection to simplify graphical data without having to resort to block mario
* Setup for modern CUDA based environments
* Adds additional default movements to action space including run and backward movement
* Uses a probability distribution when determining random movement (e.g. forward movements set to be more likely than backward)
* Significant code cleanup and updates to be more consistent with current Pytorch standards


## Installation:
> [!IMPORTANT]
> Note: nes-py requires visual C++ build tools

` python -m venv .`
` ./Scripts/Activate `
` pip install -r requirments.txt `

## Config:
` Edit config.py for all basic parameters `

## Training:
` python mario.py `

## Running Trained Agent:
Note: you need to train the agent first
` python mario.py --play`

## Other command line options:
```
  --visualize #Turns on the gamplay visuals
  --no_log #Disables logging
  --num_episodes #Shortcut to override config number of episodes
```
