import numpy
import torch


# Tensor Basic

t2 = numpy.array([1, 2, 3])
t = torch.FloatTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
'''
print(t)
print(t2)
print(t.size())
print(t.dim())
print(t[:, 1:])
'''
"""
# sum multiple
m1 = torch.cuda.FloatTensor([2, 3])
m2 = torch.cuda.FloatTensor([3])
print(m1+m2)

m1 = torch.cuda.FloatTensor([[2, 3]])
m2 = torch.cuda.FloatTensor([[3], [6]])

print(m1.matmul(m2))
print(m2.matmul(m1))
print(m1.size(), m2.size())

m1 = torch.cuda.FloatTensor([[2, 3, 4]])
m2 = torch.cuda.FloatTensor([[3], [5]])
print(m1*m2)
print(m1.mul(m2))

t = torch.cuda.FloatTensor([[2, 3],
                       [5, 8]])
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.mean(dim=-1))

print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))

print(t.max())
print(t.max(dim=0))
print(t.max(dim=0)[1])


t1 = torch.cuda.FloatTensor([[1, 2, 3],
                        [4, 5, 13],
                        [7, 14, 9],
                        [10, 11, 12]])

print(t1.argmax())
"""
"""
# View cat squeeze stack mul_ ones_like
ft = torch.FloatTensor([[[0, 1, 2],
                         [3, 4, 5]],
                        [[6, 7, 8],
                         [9, 10, 11]]])
print(ft.size())
print(ft.view([-1, 3]))
print(ft.view([-1, 3]).shape)

print(ft.view([3, -1]))
print(ft.view([-1]))
print(ft.view([-1, 1, 3]))

ft = torch.FloatTensor([[1, 2, 3]])
print(ft.shape)
print(ft.squeeze())
print(ft.squeeze().shape)

ft = torch.FloatTensor([1, 2, 3])
print(ft.shape)
print(ft.unsqueeze(0))
print(ft.unsqueeze(1))
print(ft.unsqueeze(-1))
# print(ft.unsqueeze(2))

lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
print(lt.float())

bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.float())


x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x, y], dim=0))
print(torch.cat([x, y], dim=1))

x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
print(torch.stack([x, y, z], dim=1))


x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
print(torch.ones_like(x))
print(torch.zeros_like(x))

x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2))
print(x)
x.mul_(2)
print(x)
"""
