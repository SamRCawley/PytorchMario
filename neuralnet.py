from torch import nn
import copy

class MarioNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        #Convolution network for normalized RGB image.
        self.online = nn.Sequential(
            nn.Conv3d(in_channels=input_dim[0], out_channels=16, kernel_size=(2, 4, 4), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(1, 4, 4), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 4, 4), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(1, 4, 4), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5824, 32768),
            nn.ReLU(),
            nn.Linear(32768, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        output = None
        if model == 'online':
            output =  self.online(input)
        elif model == 'target':
            output = self.target(input)
        return output
