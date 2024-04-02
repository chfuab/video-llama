import torch
import torch.nn as nn
from video_llama.common.registry import registry

@registry.register_model("lora")
class LoRA(nn.Module):
    def __init__(self, dim_A, dim_B, rank, alpha):
        self.A = nn.Linear(dim_A, rank)
        self.B = nn.Linear(rank, dim_B)
        self.alpha = alpha
        self.dim_input = self.A.in_features
        
        torch.nn.init.zeros(self.B.weight)

    def forward(self, inputs):
        assert (inputs.size()[-1] == self.dim_input.in_features), \
            "size of last dimension of inputs must be same as dim_A"
        return self.alpha * self.B(self.A(inputs))