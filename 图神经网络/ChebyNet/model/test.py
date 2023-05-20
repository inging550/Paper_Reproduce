import torch

a = torch.tensor([[0, 1.2, 0],[2, 3.1, 0],[0.5, 0, 0]])
idx = torch.nonzero(a).T  # 这里需要转置一下
data = a[idx[0],idx[1]]
coo_a = torch.sparse_coo_tensor(idx, data, a.shape)

b = torch.tensor([[0, 1.2, 0],[2, 3.1, 0],[0.5, 0, 0]])
idx = torch.nonzero(b).T  # 这里需要转置一下
data = b[idx[0],idx[1]]
coo_b = torch.sparse_coo_tensor(idx, data, b.shape)

dd = torch.mm(coo_a, coo_b)
ee = torch.sparse.mm(coo_a, coo_b)

print(dd)
print(dd.is_sparse)
print(dd.to_dense())