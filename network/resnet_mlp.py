import numpy as np
import torch
import torch.nn as nn

class GaussianActivation(nn.Module):
    def __init__(self, a=1.0):
        super(GaussianActivation, self).__init__()
        self.a = a

    def forward(self, x):
        return torch.exp(-0.5*x**2 / self.a**2)

class ResnetBlock(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_gaussian=False):
        super().__init__()

        if use_gaussian:
            self.prelu_0 = GaussianActivation()
            self.prelu_1 = GaussianActivation()
        else:
            self.prelu_0 = torch.nn.ReLU(inplace=True)
            self.prelu_1 = torch.nn.ReLU(inplace=True)
        
        self.fc_0 = torch.nn.Linear(input_size, hidden_size)
        self.fc_1 = torch.nn.Linear(hidden_size, output_size)

        self.shortcut = (
            torch.nn.Linear(input_size, output_size, bias=False)
            if input_size != output_size else None)
            

    def forward(self, x):
        residual = self.fc_1(self.prelu_1(self.fc_0(self.prelu_0(x))))
        shortcut = x if self.shortcut is None else self.shortcut(x)
        return residual + shortcut


class PosEncodeResnet(torch.nn.Module):
    def __init__(self, args, pos_size, x_size,
                hidden_size, output_size, block_num, freq_factor=np.pi, use_gaussian=False):
        """
        Args:
            pos_size: size of positional encodings
            x_size: size of input vector
            hidden_size: hidden channels
            output_size: output channels
            freq_num: how many frequency bases
            block_num: how many resnet blocks
        """
        super().__init__()

        self.args = args
        self.freq_factor = freq_factor

        input_size = (
            pos_size * (2 * self.args.freq_num + 1)
            + x_size
        )

        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.blocks = torch.nn.ModuleList(
            [ResnetBlock(hidden_size, hidden_size, hidden_size, use_gaussian=use_gaussian)
             for i in range(block_num)]
        )
        if use_gaussian:
            self.output_prelu = GaussianActivation()
        else:
            self.output_prelu = torch.nn.ReLU(inplace=True)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

    def posenc(self, x):
        freq_multiplier = (
            self.freq_factor * 2 ** torch.arange(
                                        self.args.freq_num,
                                        device=x.device
                                    )
        ).view(1, 1, 1, -1)
        x_expand = x.unsqueeze(-1)
        sin_val = torch.sin(x_expand * freq_multiplier)
        cos_val = torch.cos(x_expand * freq_multiplier)
        return torch.cat(
            [x_expand, sin_val, cos_val], -1
        ).view(x.shape[:2] + (-1,))

    def forward(self, pos_x, in_x):
        """
        Args:
            pos_x: input to be encoded with positional encodings
            in_x: input NOT to be encoded with positional encodings
        """
        x = self.posenc(pos_x)
        x = torch.cat([x, in_x], axis=-1)
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        out = self.output_layer(self.output_prelu(x))
        out = torch.cat([self.sigmoid(out[..., :-1]), self.softplus(out[..., -1:])], -1)
        return out
