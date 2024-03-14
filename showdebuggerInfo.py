import torch
print(torch.__version__)
torch.version.cuda
print(torch.cuda.is_available())

import torch
a=torch.tensor([1]).cuda()
print(a)
