"""
Author: Alexander Mehta
"""
import torch
import torch.nn as nn
from torchstat import stat
# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,num_layers=2)

    def forward(self, x):
        out, h = self.gru(x)
        return out, h

# Define the input tensor
batch_size = 1
seq_len = 128
input_size = 440
x = torch.randn(batch_size, seq_len, input_size).cuda()

# Create an instance of the GRU model
hidden_size = 256
gru_model = GRUModel(input_size, hidden_size).cuda()
from natten.flops import get_flops


from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  macs, params = get_model_complexity_info(gru_model, (128,440),as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))

