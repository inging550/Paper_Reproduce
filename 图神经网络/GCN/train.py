import torch.nn as nn
import torch.nn.functional as F
import torch
from Dataset2 import MyOwnDataset, Mydataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch_geometric
# 加载MNIST Superpixels数据集，并将其分成训练集和测试集
train_dataset = Mydataset("x_train.txt", "edge_train.txt", "y_train.txt", 60000)
test_dataset = Mydataset("x_test.txt", "edge_test.txt", "y_test.txt", 10000)
train_data_list = train_dataset.creat_dataset()
test_data_list = test_dataset.creat_dataset()
# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 64)
        self.fc1 = torch.nn.Linear(64, 512)
        self.fc2 = torch.nn.Linear(512, 10)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                nn.init.kaiming_normal_(i.weight, mode='fan_out')
                if i.bias is not None:
                    nn.init.constant_(i.bias, 0)
            elif isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, 0, 0.01)  # 全连接层初始为正态分布均值为0方差为0.01
                nn.init.constant_(i.bias, 0)  # 用单位矩阵来填充2维输入张量或变量。在线性层尽可能多的保存输入特性。

    # edge_index：边索引矩阵，每列包含两个节点的索引，表示这两个节点之间存在一条边。（2*边的数量）-> 其可以代表邻接矩阵
    # y：节点类别向量，表示每个超像素所属的数字类别。
    # x：节点特征矩阵，每行表示一个超像素的特征向量，维度为 1（特征表示为其灰度值大小）平均每张图有75个节点
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x->(600*1)
        x = F.relu(self.conv1(x, edge_index))   # x->(600*32)
        x = F.relu(self.conv2(x, edge_index))  # x->(600*64)
        x = torch_geometric.nn.global_max_pool(x, batch)   # x->(8*64)
        x = F.relu(self.fc1(x))  # x->(8*128)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)   # x->(8*10)
        return x

# 定义测试函数
def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

if __name__ == '__main__':
    # 定义超参数并训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 8
    now_lr = 0.01
    best_accuracy = 0
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=now_lr, weight_decay=5e-4)
    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
    loss_function.to(device)
    train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

    # 开始训练
    model.train()
    for epoch in range(num_epochs):
        if epoch == 2:
            now_lr = 1E-3  # 第四次迭代时学习率设置为0.0001
        if epoch == 4:
            now_lr = 1E-4
        if epoch == 6:
            now_lr = 1E-5
        for param_group in optimizer.param_groups:  # 其中的元素是2个字典；optimizer.param_groups[0]： 长度为6的字典，包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数；
            param_group['lr'] = now_lr  # 更改全部的学习率
        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(now_lr))
        total_loss = 0.
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            # loss = F.nll_loss(output, data.y)
            loss = loss_function(output, data.y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            # if i % 400 == 0:
            #     print(output)
            #     print(data.y)
            if (i + 1) % 400 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch + 1, num_epochs,
                                                                                      i + 1, len(train_loader),
                                                                                      loss.item(),
                                                                                      total_loss / (i + 1)))
        # 评估
        model.eval()
        correct = 0
        for i, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
        now_accuracy = correct / len(test_loader.dataset)
        print(now_accuracy)
        if best_accuracy < now_accuracy:
            best_accuracy = now_accuracy
            print('get best test loss %.5f' % best_accuracy)
            torch.save(model.state_dict(), 'GCN.pth')  # 保存模型参数

