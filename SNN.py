import torch
import torch.nn as nn
from block import FC_block

class SNN(nn.Module):
    def __init__(self, hyperparams):
        super(SNN, self).__init__()
        self.hyperparams = hyperparams
        if self.hyperparams[5] == 'Cifar10':
            self.hidden_size = 1500
        else:
            self.hidden_size = 500
        self.layers = nn.ModuleList()
        self.layers_size = [self.hyperparams[1], self.hidden_size, self.hyperparams[2]]
        self.len = len(self.layers_size) - 1
        self.error = None

        for i in range(self.len):
            self.layers.append(FC_block(self.hyperparams, self.layers_size[i], self.layers_size[i + 1]))

    def forward(self, input):
        for step in range(self.hyperparams[4]):
            if self.hyperparams[5] == 'MNIST':
                x = input > torch.rand(input.size()).cuda()
            elif self.hyperparams[5] == 'FashionMNIST':
                x = input > torch.rand(input.size()).cuda()
            elif self.hyperparams[5] == 'NETtalk':
                x = input.cuda()
            elif self.hyperparams[5] == 'Cifar10':
                x = input > torch.rand(input.size()).cuda()
            elif self.hyperparams[5] == 'NMNIST':
                x = input[:, :, :, :, step]
            elif self.hyperparams[5] == 'TiDigits':
                x = input[:, :, :, step]
            elif self.hyperparams[5] == 'Timit':
                x = input[:, step, :,]
            x = x.float()
            x = x.view(self.hyperparams[0], -1)
            y = x
            for i in range(self.len):
                y = self.layers[i](y)

        outputs = self.layers[-1].sumspike / self.hyperparams[4]

        return outputs