import torch
import numpy as np
from torch import nn
from torch.nn import init
from torch.nn import functional

class ConvGru(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 1, kernel_size: tuple[int, ...] | int = (3, 3),
                 stride: tuple[int, ...] | int = (1, 1), padding: tuple[int, ...] | int = (1, 1)):
        super( ).__init__( )

        # initialize update, reset, and output gates
        self.batch_norms = nn.ModuleList( )
        self.batch_norms.append(nn.BatchNorm2d(input_size + hidden_size))
        self.batch_norms.append(nn.BatchNorm2d(input_size + hidden_size))

        self.reset_gates = nn.ModuleList( )
        self.reset_gates.append(nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding))
        self.reset_gates.append(nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding))

        self.update_gates = nn.ModuleList( )
        self.update_gates.append(nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding))
        self.update_gates.append(nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding))

        self.output_gates = nn.ModuleList( )
        self.output_gates.append(nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding))
        self.output_gates.append(nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding))

        init.orthogonal_(self.reset_gates[0].weight); init.zeros_(self.reset_gates[0].bias)
        init.orthogonal_(self.reset_gates[1].weight); init.zeros_(self.reset_gates[1].bias)

        init.orthogonal_(self.update_gates[0].weight); init.zeros_(self.update_gates[0].bias)
        init.orthogonal_(self.update_gates[1].weight); init.zeros_(self.update_gates[1].bias)

        init.orthogonal_(self.output_gates[0].weight); init.zeros_(self.output_gates[0].bias)
        init.orthogonal_(self.output_gates[1].weight); init.zeros_(self.output_gates[1].bias)

        self.prev_states = None


    def forward(self, input: torch.Tensor):
        # initialize previous state if there is none
        if self.prev_states is None:
            self.prev_states = list( )
            self.prev_states.append(torch.zeros_like(input))
            self.prev_states.append(functional.interpolate(self.prev_states[-1], scale_factor = 2))

        output = None
        for idx in range(2):
            stacked_inputs = torch.cat((input, self.prev_states[idx]), dim = 1)

            # normalize stacked inputs
            stacked_inputs = self.batch_norms[idx](stacked_inputs)

            # input concatenated features to update and reset gates
            reset = functional.sigmoid(self.reset_gates[idx](stacked_inputs))
            update = functional.sigmoid(self.update_gates[idx](stacked_inputs))

            # determine the candidate new state
            candidate_new_state = functional.tanh(self.output_gates[idx](torch.cat((input, self.prev_states[idx] * reset), dim = 1)))
            output = (self.prev_states[idx] * (1 - update)) + ((candidate_new_state) * update)

            # save hidden state
            self.prev_states[idx] = output.clone( ).detach( )
            
            # normalize output via sigmoid
            # output = functional.sigmoid(output * torch.exp(torch.tensor(1)))
            output = (output - output.min( )) / (output.max( ) - output.min( ))
            if idx != 1: input = functional.interpolate(output, scale_factor = 2, mode = "bicubic")
        return output
