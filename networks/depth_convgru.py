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

        self.scale = 4
        self.prev_states = None


    def forward(self, input):        
        # initialize previous states if there are none
        if self.prev_states is None:
            self.prev_states = list( )
            for scale in range(self.scale):
                self.prev_states.append(torch.zeros_like(input[scale]))

        outputs = list( )
        for scale in range(self.scale):
            # concatenate input at current scale and previous state at current scale
            stacked_inputs = torch.cat((input[scale], self.prev_states[scale]), dim = 1)

            # input concatenated features to update and reset gates
            update = functional.sigmoid(self.update_gate(stacked_inputs))
            reset = functional.sigmoid(self.reset_gate(stacked_inputs)); del stacked_inputs

            # determine the candidate new state at current scale
            candidate_new_state = functional.tanh(self.output_gate(torch.cat([input[scale], self.prev_states[scale] * reset], dim = 1))); del reset

            output = self.prev_states[scale] * (1 - update) + candidate_new_state * update; del update

            outputs.append(output)
        
        # save hidden state
        self.prev_states = outputs.copy( )
        for idx in range(len(self.prev_states)):
            self.prev_states[idx] = self.prev_states[idx].detach( )
        
        # aggregate depth outputs by resizingg and taking the average of the sum of depths
        H = 192 // (2 ** (self.scale - 2)); W = 640 // (2 ** (self.scale - 2))
        for idx in range(self.scale - 1):
            outputs[idx] = functional.interpolate(outputs[idx], size = (H, W), mode = "bicubic")
            outputs[idx + 1] += outputs[idx]
            H *= 2; W *= 2
        return torch.clamp(outputs[-1] / self.scale, min = 0, max = 1)  # clamp to [0, 1]
        # return functional.sigmoid(outputs[-1] / self.scale)
