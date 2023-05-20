import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Dataset, Data, download_url
from torchvision.transforms import ToTensor
# from torch_geometric.data import InMemoryDataset

class Mydataset(Dataset):
    def __init__(self, x_data, edge_data, y_data, sample_num):
        super(Mydataset, self).__init__()
        self.sample_num = sample_num
        self.x_info = torch.zeros((sample_num*81, 3))
        self.edge_info = []
        self.y_info = torch.zeros((sample_num), dtype=torch.long)
        transform1 = ToTensor()
        self.total_data = []
        i = 0  # 表示现在在第几个样本
        with open(x_data, 'r') as f:
            for line in f.readlines():
                samplei_info = line.strip().split()
                self.x_info[i*81:(i+1)*81, :] = transform1(np.array([int(float(j)) for j in samplei_info]).reshape((81, 3)))
                i += 1
        with open(edge_data, 'r') as f:
            for line in f.readlines():
                samplei_info = line.strip().split()
                self.edge_info.append([int(float(j)) for j in samplei_info])
        i = 0
        with open(y_data, 'r') as f:
            for line in f.readlines():
                samplei_info = line.strip()
                self.y_info[i] = int(float(samplei_info))
                i += 1

    def creat_dataset(self):
        for idx in range(self.sample_num):
            sample_edge_info = torch.cat((torch.Tensor(self.edge_info[2 * idx]).type(torch.int64).unsqueeze(0),
                                          torch.Tensor(self.edge_info[2 * idx + 1]).type(torch.int64).unsqueeze(0)),
                                         dim=0)
            data = Data(x=self.x_info[idx * 81: (idx + 1) * 81], edge_index=sample_edge_info, y=self.y_info[idx])
            self.total_data.append(data)
        return self.total_data

    def get(self, idx: int) -> Data:
        sample_edge_info = torch.cat((torch.Tensor(self.edge_info[2 * idx]).type(torch.int64).unsqueeze(0), torch.Tensor(self.edge_info[2 * idx + 1]).type(torch.int64).unsqueeze(0)), dim=0)
        data = Data(x=self.x_info[idx*81: (idx+1)*81], edge_index=sample_edge_info, y=self.y_info[idx])
        return data

    def len(self) -> int:
        return self.sample_num


# if __name__ == '__main__':
#     train_dataset = Mydataset(x_data="x_train.txt", edge_data="edge_train.txt", y_data="y_train.txt", sample_num=60000)
#     test_dataset = Mydataset(x_data="x_test.txt", edge_data="edge_test.txt", y_data="y_test.txt", sample_num=10000)