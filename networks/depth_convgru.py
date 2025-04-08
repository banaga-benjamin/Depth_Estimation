import torch
import numpy as np
from torch import nn
from torch.nn import init
from torch.nn import functional

class ConvGru(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 1, kernel_size: tuple[int, ...] | int = (3, 3),
                 stride: tuple[int, ...] | int = (1, 1), padding: tuple[int, ...] | int = (1, 1)):
        super( ).__init__( )

        # initialize batchnorm, update, reset, and output gates
        self.batch_norm = nn.BatchNorm2d(input_size + hidden_size)
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding)
        self.output_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding)

        init.orthogonal_(self.reset_gate.weight); init.zeros_(self.reset_gate.bias)
        init.orthogonal_(self.update_gate.weight); init.zeros_(self.update_gate.bias)
        init.orthogonal_(self.output_gate.weight); init.zeros_(self.output_gate.bias)

        # initialize previous state
        self.prev_state = None


    def forward(self, input: torch.Tensor):
        # initialize previous state if there is none
        if self.prev_state is None: self.prev_state = torch.zeros_like(input)

        # stack inputs and normalize
        stacked_inputs = self.batch_norm(torch.cat([input, self.prev_state], dim = 1))

        # input concatenated features to update and reset gates
        reset = functional.sigmoid(self.reset_gate(stacked_inputs))
        update = functional.sigmoid(self.update_gate(stacked_inputs))

        # determine the candidate new state
        candidate_new_state = functional.tanh(self.output_gate(torch.cat((input, self.prev_state * reset), dim = 1)))
        output = (self.prev_state * (1 - update)) + ((candidate_new_state) * update)

        # save hidden state
        self.prev_state = output.clone( ).detach( )
        
        # normalize final output by passing to sigmoid
        return functional.sigmoid(output)
