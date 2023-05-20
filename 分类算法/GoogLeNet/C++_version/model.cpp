#include "model.h"

// ���캯����������ṹ
GoogLeNetImpl::GoogLeNetImpl(int NUM_CLASS, bool init_weight)
{
	// ������ȡ���ֵ�����ṹ
	block1 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3)),  // ��������
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),          // ReLu�����
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)),   // �������ػ���

		torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 1).stride(1)),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 3).stride(1).padding(1)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1).ceil_mode(false))
	);
	block2 = torch::nn::Sequential(
		Inception(input1),
		Inception(input2),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2)),
		Inception(input3)
	);
	block3 = torch::nn::Sequential(
		Inception(input4),
		Inception(input5),
		Inception(input6)
	);
	block4 = torch::nn::Sequential(
		Inception(input7),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1).ceil_mode(false)),
		Inception(input8),
		Inception(input9),
		torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1,1}))
	);
	end_block = torch::nn::Sequential(
		//torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)),
		torch::nn::Dropout(0.4),
		torch::nn::Linear(1024, NUM_CLASS)
	);

	// ��libtorch�ж�������綼Ҫע��һ��
	register_module("block1", block1);
	register_module("block2", block2);
	register_module("block3", block3);
	register_module("block4", block4);
	register_module("end_block", end_block);

	//if (init_weight)
	//{
	//	define_weight();
	//}
}

//ǰ�򴫲�����
torch::Tensor GoogLeNetImpl::forward(torch::Tensor& x)
{	
	x = block1->forward(x);
	x = block2->forward(x);
	x = block3->forward(x);
	x = block4->forward(x);
	x = torch::flatten(x, 1);
	x = end_block->forward(x);
	return x;
}

//��ʼ��Ȩ�ز���
void GoogLeNetImpl::define_weight()
{
	for (auto m : this->modules(false))
	{
		if (m->name() == "torch::nn::Conv2dImpl")  // ��ʼ����������
		{
			printf("init the conv2d parameters.\n");
			auto spConv2d = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(m);
			spConv2d->reset_parameters();
			// Kaiming He �����Ȩ�س�ʼ������
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
		else if (m->name() == "torch::nn::LinearImpl")   // ��ʼ��ȫ���Ӳ����
		{
		printf("init the Linear parameters.\n");
		  auto spLinear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(m);
		  torch::nn::init::normal_(spLinear->weight,0,0.01);
		  torch::nn::init::constant_(spLinear->bias, 0);
		}
	}

}


InceptionImpl::InceptionImpl(int input[7])
{
	path1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(input[0], input[1], 1).stride(1));
	path2 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(input[0],input[2],1).stride(1)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(input[2], input[3], 3).stride(1).padding(1)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true))
	);
	path3 = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(input[0], input[4], 1).stride(1)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(input[4], input[5], 5).stride(1).padding(2)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true))
	);
	path4 = torch::nn::Sequential(
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(1).padding(1)),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(input[0], input[6], 1).stride(1))
	);
	// ע��һ��
	register_module("path1", path1);
	register_module("path2", path2);
	register_module("path3", path3);
	register_module("path4", path4);
}

torch::Tensor InceptionImpl::forward(torch::Tensor x)
{
	x1 = torch::nn::functional::relu( path1->forward(x));
	x2 = path2->forward(x);
	x3 = path3->forward(x);
	x4 = torch::nn::functional::relu(path4->forward(x));
	return torch::cat({ x1,x2,x3,x4 }, 1);
}
