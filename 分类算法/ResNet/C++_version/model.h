#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>


class ResNetImpl : public torch::nn::Module
{
public:
	ResNetImpl(int NUM_CLASS);  // NUM_CLASS为数据集的类别总数，init_weight为是否初始化权重参数
	torch::Tensor forward(torch::Tensor x);  // 前向传播函数
	torch::nn::Sequential make_layer(int in_channel, int out_channel, int stack_num, int stride);
	void define_weight();
private:
	// 定义网络结构
	torch::nn::Sequential conv1{ nullptr };
	torch::nn::Sequential conv2_3{nullptr};  
	torch::nn::Sequential conv3_4{nullptr};
	torch::nn::Sequential conv4_23{ nullptr };
	torch::nn::Sequential conv5_3{ nullptr };
	torch::nn::Sequential end_layer{ nullptr };
};
TORCH_MODULE(ResNet);

class BottleneckImpl : public torch::nn::Module
{
public:
	BottleneckImpl(int in_channel, int out_channel, int stride, bool if_downsample);
	torch::Tensor forward(torch::Tensor x);
// Bottleneck的网络结构
private:
	torch::nn::Sequential down_sample{ nullptr };
	torch::nn::Sequential residual{ nullptr };
// 定义前向传播的一些参数
private:
	torch::Tensor identity_tensor;
	torch::Tensor residual_tensor;
};
TORCH_MODULE(Bottleneck);
