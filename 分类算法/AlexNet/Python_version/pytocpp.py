# 此文将将pytorch训练的模型文件转变为Libtorch可以用的模型文件（Pytorch和Libtorch版本需要一致）
import torch
# 加载模型
from model import AlexNet

model = AlexNet(NUM_CLASS=10, init_weight=True)
checkpoint = torch.load(r"AlexNet.pth")
model.load_state_dict(checkpoint)
model.cuda()
model.eval()

# 向模型中输入数据以得到模型参数
example = torch.rand(1, 3, 224, 224).cuda()
traced_script_module = torch.jit.trace(model, example)

# 保存模型
traced_script_module.save("AlexNet.pt")