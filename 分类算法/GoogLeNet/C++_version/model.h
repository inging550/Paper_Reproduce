#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>


class GoogLeNetImpl : public torch::nn::Module
{
public:
	GoogLeNetImpl(int NUM_CLASS, bool init_weight);  // NUM_CLASSΪ���ݼ������������init_weightΪ�Ƿ��ʼ��Ȩ�ز���
	torch::Tensor forward(torch::Tensor& x);  // ǰ�򴫲�����
	void define_weight();  // ��ʼ��Ȩ�ز���
private:
	// ��������ṹ
	torch::nn::Sequential block1{nullptr};  
	torch::nn::Sequential block2{nullptr};  
	torch::nn::Sequential block3{ nullptr };
	torch::nn::Sequential block4{ nullptr };
	torch::nn::Sequential end_block{ nullptr };
// �������Inception�ṹ
private:
	int input1[7] = {192,64,96,128,16,32,32};
	int input2[7] = {256,128,128,192,32,96,64 };
	int input3[7] = {480,192,96,208,16,48,64 };
	int input4[7] = {512,160,112,224,24,64,64 };
	int input5[7] = {512,128,128,256,24,64,64 };
	int input6[7] = {512,112,144,288,32,64,64 };
	int input7[7] = {528,256,160,320,32,128,128 };
	int input8[7] = {832,256,160,320,32,128,128 };
	int input9[7] = {832,384,192,384,48,128,128 };

};
TORCH_MODULE(GoogLeNet);

class InceptionImpl : public torch::nn::Module
{
public:
	InceptionImpl(int m_input[7]);
	torch::Tensor forward(torch::Tensor x);
// ����inception�Ľṹ��
private:
	torch::nn::Conv2d path1{ nullptr };
	torch::nn::Sequential path2{ nullptr };
	torch::nn::Sequential path3{ nullptr };
	torch::nn::Sequential path4{ nullptr };
// ����ǰ�򴫲���һЩ����
private:
	torch::Tensor x1;
	torch::Tensor x2;
	torch::Tensor x3;
	torch::Tensor x4;
};
TORCH_MODULE(Inception);
