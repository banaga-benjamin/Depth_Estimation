import torch
from torch import nn
from torch.nn import init
from torch.nn import functional


class ConvGru(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), device = 'cpu'):
        super( ).__init__( )

        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding).to(device)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding).to(device)
        self.output_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, stride = stride, padding = padding).to(device)
        
        init.orthogonal_(self.reset_gate.weight); init.constant_(self.reset_gate.bias, 0.0)
        init.orthogonal_(self.update_gate.weight); init.constant_(self.update_gate.bias, 0.0)
        init.orthogonal_(self.output_gate.weight); init.constant_(self.output_gate.bias, 0.0)

        self.scale = 4
        self.device = device
        self.prev_states = None

    def forward(self, input):        
        # initialize previous states if there are none
        if self.prev_states is None:
            self.prev_states = list( )
            for scale in range(self.scale):
                self.prev_states.append(torch.zeros_like(input[scale], device = self.device))

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
        return outputs


# for debugging
# if __name__ == "__main__":
#    device = 'cuda' if torch.cuda.is_available( ) else 'cpu'
  
#    H = 192; W = 640
#    input = [torch.rand(1, 1, H // 8, W // 8, device = device),
#             torch.rand(1, 1, H // 4, W // 4, device = device),
#             torch.rand(1, 1, H // 2, W // 2, device = device),
#             torch.rand(1, 1, H, W, device = device)]
   

#    model = ConvGru(device = device); outputs = model(input)
#    for output in outputs: print(output.shape)
