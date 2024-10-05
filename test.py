import torch
import torch.nn as nn

class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.attr = torch.nn.Parameter(torch.rand(5), requires_grad=False)


a = A()
b = {key: value.detach().clone() for key, value in a.state_dict().items()}
print(b)
a.attr += 1
print(b)
print(a.attr)