import torch
import torch.nn as nn

### 
# MODEL SPECIFICATIONS TO BE MESSED WITH
# current is not good; just fast
###

class CNNClassifer(nn.Module):
    '''
    binary classification for 64 x 64 patch 
    '''

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2), # 64 x 64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 32 x 32

            nn.Conv2d(16, 32, 5, padding=2), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 16 x 16

            nn.Conv2d(32, 32, 3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 8 x 8

            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 4 x 4
        )
        
        self.decoder = nn.Sequential( # currently 1 hidden layer
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.decoder(x)
        return x