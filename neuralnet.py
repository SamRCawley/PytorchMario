from torch import nn
import copy

class MarioNet(nn.Module):
    '''mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        #initial kernel size is 16 / 4 which is the resized dimension of a sprite
        """self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(245760, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )"""
        #LSTM with 256x240 (flattened) input layer
        self.online = nn.LSTM(61440, 50, 4)
        self.target = copy.deepcopy(self.online)
        #Linear layer mapping LSTM output to action space
        self.online_to_action = nn.Linear(50, output_dim)
        self.target_to_action = nn.Linear(50, output_dim)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False
        for p in self.target_to_action.parameters():
            p.requires_grad = False

    def forward(self, input, net_tuple, model):
        output, hidden = None, None
        if model == 'online':
            if net_tuple is not None:
                output, hidden =  self.online(input, net_tuple)
            else:
                output, hidden = self.online(input)
            output = self.online_to_action(output[-1])
        elif model == 'target':
            if net_tuple is not None:
                output, hidden = self.target(input, net_tuple)
            else:
                output, hidden = self.target(input)
            output = self.target_to_action(output[-1])
        return output, hidden
