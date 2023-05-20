import torch.types
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optioms
from model import AlexNet
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)   # 打印当前使用的设备（GPU(cuda:0)还是CPU）
num_epochs = 10  # 设定的迭代次数上限
learning_rate = 0.001  # 初始学习率
best_test_loss = 20.0
# 1、定义数据集
transform = {
    # ToTensor()能够把像素的值域范围从0-255变换到0-1之间，
    # 而后面的transform.Normalize()则把0-1变换到(-1,1).
    'train': transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.RandomHorizontalFlip(),  # 随机翻转
                                 transforms.ToTensor(),  # 转换为张量
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化处理
                                 ]),
    'test': transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}
# 数据集的路径根据实际情况修改
train_dataset = datasets.ImageFolder(root='F:/CCCCCProject/DATASET/TRAIN', transform=transform['train'])
test_dataset = datasets.ImageFolder(root='F:/CCCCCProject/DATASET/TEST', transform=transform['test'])

# 参数shuffle表示是否打乱数据集
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 2、定义网络结构并设置为CUDA
net = AlexNet(NUM_CLASS=10, init_weight=True)  # NUM_CLASS为当前数据集的类别总数
net.to(device)

# 打印模型信息
for k, v in net.named_parameters():
    print(k)


# 3、定义损失函数及优化器，损失函数设置为CUDA
loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optioms.SGD(params=net.parameters(), lr=learning_rate)  # SGD随机梯度下降
loss_function.to(device)

# 4、开始训练
for epoch in range(num_epochs):
    net.train()  # 网络有Dropout，BatchNorm层时一定要加
    if epoch == 4:
        learning_rate = 0.0001  # 第四次迭代时学习率设置为0.0001
    if epoch == 6:
        learning_rate = 0.00001
    for param_group in optimizer.param_groups:   # 其中的元素是2个字典；optimizer.param_groups[0]： 长度为6的字典，包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数；
                                                # optimizer.param_groups[1]： 好像是表示优化器的状态的一个字典；
        param_group['lr'] = learning_rate      # 更改全部的学习率
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    total_loss = 0.
    net.eval()
    for i, (images, target) in enumerate(train_loader):  # image为图片，target为其对应的标签（类别名）
        images, target = images.cuda(), target.cuda()  # 设置为CUDA
        pred = net(images)    # 图片输入网络得到预测结果
        loss = loss_function(pred, target)  # 将预测结果与实际标签比对（计算两者之间的损失值）
        total_loss += loss.item()

        optimizer.zero_grad()   # 将梯度归零，有助于梯度下降
        loss.backward()    # 反向传播 计算梯度
        optimizer.step()   # 根据梯度 更新模型参数
        if (i + 1) % 5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch + 1, num_epochs,
                                                                                 i + 1, len(train_loader), loss.item(), total_loss / (i + 1)))
    validation_loss = 0.0
    for i, (images, target) in enumerate(test_loader):  # 导入dataloader 说明开始训练了  enumerate 建立一个迭代序列
        images, target = images.cuda(), target.cuda()
        pred = net(images)    # 将图片输入
        loss = loss_function(pred, target)
        validation_loss += loss.item()   # 累加loss值  （固定搭配）
    validation_loss /= len(test_loader)  # 计算平均loss
    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(net.state_dict(), 'AlexNet.pth')  # 保存模型参数



