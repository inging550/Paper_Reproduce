#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>


class ResNetImpl : public torch::nn::Module
{
public:
	ResNetImpl(int NUM_CLASS);  // NUM_CLASSΪ���ݼ������������init_weightΪ�Ƿ��ʼ��Ȩ�ز���
	torch::Tensor forward(torch::Tensor x);  // ǰ�򴫲�����
	torch::nn::Sequential make_layer(int in_channel, int out_channel, int stack_num, int stride);
	void define_weight();
private:
	// ��������ṹ
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
// Bottleneck������ṹ
private:
	torch::nn::Sequential down_sample{ nullptr };
	torch::nn::Sequential residual{ nullptr };
// ����ǰ�򴫲���һЩ����
private:
	torch::Tensor identity_tensor;
	torch::Tensor residual_tensor;
};
TORCH_MODULE(Bottleneck);
