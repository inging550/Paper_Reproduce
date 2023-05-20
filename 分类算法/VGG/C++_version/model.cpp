#include "model.h"

// ���캯����������ṹ
VGG16Impl::VGG16Impl(int NUM_CLASS, bool init_weight)
{
	// ������ȡ���ֵ�����ṹ
	features = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1)),  // ��������
		torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),  // ��������
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),          // ReLu�����
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),   // �������ػ���

		torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride({2,2})),

		torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride({ 2,2 })),

		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride({ 2,2 })),

		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride({ 2,2 }))
		);

	// ���Է��ಿ�ֵ�����ṹ
	classifier = torch::nn::Sequential(
		torch::nn::Linear(torch::nn::LinearOptions(512*7*7, 4096)),     // ����ȫ���Ӳ�
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),
		torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),
		torch::nn::Linear(torch::nn::LinearOptions(4096, NUM_CLASS))    // NUM_CLASS�����Լ������ݼ������������
	 );
	// ��libtorch�ж�������綼Ҫע��һ��
	features = register_module("features", features);
	classifier = register_module("classifier", classifier);
	if (init_weight)
	{
		define_weight();
	}
}

//ǰ�򴫲�����
torch::Tensor VGG16Impl::forward(torch::Tensor x)
{	
	x = features->forward(x);
	x = torch::flatten(x, 1);
	x = classifier->forward(x);
	return x;
}

//��ʼ��Ȩ�ز���
void VGG16Impl::define_weight()
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

