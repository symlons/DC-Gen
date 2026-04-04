from dit import DiT_L_1
import torch

model = DiT_L_1()

t = torch.randint(0, 2, (2,))
y = t
x = torch.randn(2, 32, 16, 16, 16)
out = model(x, t, y)
print(out.shape)
