import torch

x = torch.randn((5, 5), requires_grad=True)
y = torch.randn((5, 5), requires_grad=True)
z = torch.randn((5, 5), requires_grad=True)

u = x + y
 # u.requires_grad = False
print(u)
v = u * z

w = torch.sum(v)

print(w)


w.backward()

print(x.grad)


# v.backward()    # this will result in error!
