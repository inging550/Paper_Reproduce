#include "model.h"

// 构造函数定义网络结构
ResNetImpl::ResNetImpl(int NUM_CLASS)
{
	// 定义网络结构
	conv1 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3)),
		torch::nn::BatchNorm2d(64),
		torch::nn::ReLU(true),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1))
	);
	conv2_3 = make_layer(64, 64, 3, 1);
	conv3_4 = make_layer(256, 128, 4, 2);
	conv4_23 = make_layer(512, 256, 23, 2);
	conv5_3 = make_layer(1024, 512, 3, 2);
	end_layer = torch::nn::Sequential(
		torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1,1})),
		torch::nn::Flatten(),
		torch::nn::Linear(2048, NUM_CLASS)
	);
	// 在libtorch中定义的网络都要注册一下
	register_module("conv1", conv1);
	register_module("conv2_3", conv2_3);
	register_module("conv3_4", conv3_4);
	register_module("conv4_23", conv4_23);
	register_module("conv5_3", conv5_3);
	register_module("end_layer", end_layer);
	define_weight();
}

//前向传播函数
torch::Tensor ResNetImpl::forward(torch::Tensor x)
{	
	x = conv1->forward(x);
	x = conv2_3->forward(x);
	x = conv3_4->forward(x);
	x = conv4_23->forward(x);
	x = conv5_3->forward(x);
	x = end_layer->forward(x);
	return x;
}

torch::nn::Sequential ResNetImpl::make_layer(int in_channel, int out_channel, int stack_num, int stride)
{
	torch::nn::Sequential block;
	block->push_back(Bottleneck(in_channel,out_channel,stride,true));
	for (int i = 1; i < stack_num; i++)
	{
		block->push_back(Bottleneck(out_channel*4, out_channel,1,false));
	}
	return block;
}

BottleneckImpl::BottleneckImpl(int in_channel, int out_channel, int stride, bool if_downsample)
{
	// 三目运算符 速度更快
	down_sample = if_downsample == true ? torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, out_channel * 4, 1).stride(stride)),
		torch::nn::BatchNorm2d(out_channel * 4)
	) : torch::nn::Sequential();

	residual = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, out_channel, 1).stride(1)),
		torch::nn::BatchNorm2d(out_channel),
		torch::nn::ReLU(true),

		torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channel, out_channel, 3).stride(stride).padding(1)),
		torch::nn::BatchNorm2d(out_channel),
		torch::nn::ReLU(true),

		torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channel, out_channel*4, 1).stride(1)),
		torch::nn::BatchNorm2d(out_channel*4)
	);
	// 注册一下
	register_module("down_sample", down_sample);
	register_module("residual", residual);
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x)
{
	residual_tensor = residual->forward(x);
	identity_tensor = down_sample->is_empty() ? x : down_sample->forward(x);
	return torch::nn::functional::relu(residual_tensor += identity_tensor, true);
}

void ResNetImpl::define_weight()
{
	for (auto m : this->modules(false))
	{
		if (m->name() == "torch::nn::Conv2dImpl")  // 初始化卷积层参数
		{
			printf("init the conv2d parameters.\n");
			auto spConv2d = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(m);
			spConv2d->reset_parameters();
			// Kaiming He 创造的权重初始化方法
			torch::nn::init::kaiming_normal_(spConv2d->weight, 0.0, torch::kFanOut, torch::kReLU);
			if (spConv2d->options.bias())
				torch::nn::init::constant_(spConv2d->bias, 0);
		}
		//else if (m->name() == "torch::nn::BatchNorm2dImpl")
		//{
		//	printf("init the batchnorm2d parameters.\n");
		//	auto spBatchNorm2d = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(m);
		//	torch::nn::init::constant_(spBatchNorm2d->weight, 1);
		//	torch::nn::init::constant_(spBatchNorm2d->bias, 0);
		//}
		else if (m->name() == "torch::nn::LinearImpl")   // 初始化全连接层参数
		{
			printf("init the Linear parameters.\n");
			auto spLinear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(m);
			torch::nn::init::normal_(spLinear->weight, 0, 0.01);
			torch::nn::init::constant_(spLinear->bias, 0);
		}
	}

}
