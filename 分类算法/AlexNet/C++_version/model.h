#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>


class AlexNetImpl : public torch::nn::Module
{
public:
	AlexNetImpl(int NUM_CLASS, bool init_weight);  // NUM_CLASSΪ���ݼ������������init_weightΪ�Ƿ��ʼ��Ȩ�ز���
	torch::Tensor forward(torch::Tensor x);  // ǰ�򴫲�����
	void define_weight();  // ��ʼ��Ȩ�ز���
private:
	torch::nn::Sequential features{nullptr};  // ������ȡ���ֵ�����ṹ
	torch::nn::Sequential classifier{ nullptr };   // ���Է��ಿ�ֵ�����ṹ
};
TORCH_MODULE(AlexNet);
