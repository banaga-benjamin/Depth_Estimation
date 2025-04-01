import torch
from torch import nn
from torch.nn import init
from torch.nn import functional


class ConvGru(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)):
        super( ).__init__( )

        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding)
        self.output_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding)
        
        init.orthogonal_(self.reset_gate.weight); init.constant_(self.reset_gate.bias, 0.0)
        init.orthogonal_(self.update_gate.weight); init.constant_(self.update_gate.bias, 0.0)
        init.orthogonal_(self.output_gate.weight); init.constant_(self.output_gate.bias, 0.0)

        self.prev_state = None


    def forward(self, input):
        # initialize previous state if there is none
        if self.prev_state is None: self.prev_state = torch.zeros_like(input)

        # concatenate input and previous state
        stacked_inputs = torch.cat((input, self.prev_state), dim = 1)

        # input concatenated features to update and reset gates
        update = functional.sigmoid(self.update_gate(stacked_inputs))
        reset = functional.sigmoid(self.reset_gate(stacked_inputs)); del stacked_inputs

        # determine the candidate new state
        candidate_new_state = functional.tanh(self.output_gate(torch.cat((input, self.prev_state * reset), dim = 1))); del reset
        output = (self.prev_state * (1 - update)) + ((candidate_new_state) * update); del update

        self.prev_state = output.clone( ).detach( )

        return torch.clamp(output, min = 0, max = 1)  # clamp to [0, 1]
