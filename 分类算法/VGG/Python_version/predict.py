import cv2
import torch
from model import VGG16
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# 初始化一些参数
img_root = "F:/CCCCCProject/AlexNet/Project1/DATASET/TRAIN/6/6_00022.bmp"  # 需要预测的图片路径
net = VGG16(NUM_CLASS=10, init_weight=False)  # 给神经网络实例化对象
net.load_state_dict(torch.load("AlexNet.pth"))  # 导入权重参数
net.eval()

# 各类别名字，根据自己使用的数据集更改
labels_name = ['数字0', '数字1', '数字2', '数字3', '数字4', '数字5', '数字6', '数字7', '数字8', '数字9']


transform = transforms.Compose([transforms.ToTensor(),  # 转换为张量
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 用于绘制预测的混淆矩阵
class DrawConfusionMatrix:
    def __init__(self, labels_name, normalize=True):
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, labels, predicts):
        for predict, label in zip(labels, predicts):
            self.matrix[label, predict] += 1

    def getMatrix(self, normalize=True):
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算
            for i in range(self.num_classes):
                self.matrix[i] = (self.matrix[i] / per_sum[i])  # 百分比转换
            self.matrix = np.around(self.matrix, 2)  # 保留2位小数点
            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN，将其设为0
        return self.matrix

    def drawMatrix(self):
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix, cmap="BuGn")  # 仅画出颜色格子，没有值
        plt.title("AlexNet ConfusionMatrix")  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels_name)  # y轴标签
        plt.xticks(range(self.num_classes), self.labels_name, rotation=45)  # x轴标签

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  # 写值

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        plt.colorbar()  # 色条
        plt.savefig('./ConfusionMatrix.png', bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
        plt.show()

# 进行预测
class Predict:
    def __init__(self, image_root, net1): # image_root->需要预测的图片路径，net1->网络结构
        self.img_root = image_root
        self.model = net1

    def result(self):
        img = cv2.imread(img_root)  # opencv读取图片
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR->RGB
        img = cv2.resize(img, (224, 224))  # 将图片尺寸变为224*224
        img = transform(img)   # 将图片对应的像素矩阵变为张量并且标准化  size=(3,224,224))
        img = torch.unsqueeze(img, 0)  # 执行完后尺寸为(1,3,224,224)
        result = self.model(img)  # 将图片输入神经网络得到结果
        return result


if __name__ == "__main__":
    Pre = Predict(img_root, net)
    result = Pre.result()
    _, class1 = torch.max(result, 1)
    print("此图片预测的类别为:", labels_name[class1.item()])
    # test_dataset = datasets.ImageFolder(root='DATASET/TEST', transform=transform)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # drawconfusionmatrix = DrawConfusionMatrix(labels_name=labels_name)  # 实例化
    # for index, (imgs, labels) in enumerate(test_loader):
    #     labels_pd = net(imgs)
    #     predict_np = np.argmax(labels_pd.cpu().detach().numpy(), axis=-1)  # array([0,5,1,6,3,...],dtype=int64)
    #     labels_np = labels.numpy()  # array([0,5,0,6,2,...],dtype=int64)
    #     drawconfusionmatrix.update(labels_np, predict_np)  # 将新批次的predict和label更新（保存）
    # drawconfusionmatrix.drawMatrix()  # 根据所有predict和label，画出混淆矩阵
    # confusion_mat = drawconfusionmatrix.getMatrix()  # 你也可以使用该函数获取混淆矩阵(ndarray)

