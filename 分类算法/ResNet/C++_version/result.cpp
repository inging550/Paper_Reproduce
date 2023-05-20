# include "result.h"


// ���캯�����ڳ�ʼ��һЩ����
// ����GitHub�е�˵�������޸�
Result::Result() : device_type(torch::kCUDA), img_root("F:/CCCCCProject/DATASET/TEST/4/4_00122.bmp"), NUM_CLASS(10), batch_size(8)
{
}

// �˺���Ŀ��Ϊ��ѵ��;�и���ѧϰ��
void Result::updata_learn_rate(torch::optim::SGD &optimizer, double alpha)
{
	for (auto& pg : optimizer.param_groups())
	{
		if (pg.has_options())
		{
			auto& options = pg.options();
			static_cast<torch::optim::SGDOptions&>(pg.options()).lr() = alpha;
		}
	}
}


void Result::train()
{
	// 1���������ݼ�
	auto train_dataset = dataSetClc("F:\\CCCCCProject\\DATASET\\TRAIN", ".bmp")
		.map(torch::data::transforms::Normalize<torch::Tensor>({ 0.5, 0.5, 0.5 }, { 0.5,0.5,0.5 }))
		.map(torch::data::transforms::Stack<>());   // ���ݼ��Զ��� bmpΪ��׺����ͼƬ�ĺ�׺��Ҳ����Ϊ��������:JPG,PNG�ȣ�
	auto test_dataset = dataSetClc("F:\\CCCCCProject\\DATASET\\TEST", ".bmp")
		.map(torch::data::transforms::Normalize<torch::Tensor>({ 0.5, 0.5, 0.5 }, { 0.5,0.5,0.5 }))
		.map(torch::data::transforms::Stack<>());   // ���ݼ��Զ��� bmpΪ��׺����ͼƬ�ĺ�׺��Ҳ����Ϊ��������:JPG,PNG�ȣ�



	auto num_train = train_dataset.size();
	auto num_test = test_dataset.size();
	std::cout << "ѵ��������Ϊ�� " << num_train.value() << std::endl;
	std::cout << "���Լ�����Ϊ�� " << num_test.value() << std::endl;
	auto train_dataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);   // �����ݼ����
	auto test_dataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(test_dataset), batch_size);  // batch_size = 2
	//auto train_dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), batch_size);   // �����ݼ����

	// 2����������ṹ ����ʼ��Ȩ�ز���
	ResNet m_net(NUM_CLASS);  //NUM_CLASSΪ���ݼ��������
	m_net->to(device_type);
	std::cout << device_type << std::endl;
	// ��ӡģ�ͽṹ
	//std::cout << "ģ�ͽṹ����" << std::endl;
	//for (auto& i : m_net->named_parameters())
	//{
	//	std::cout << i.key() << std::endl;
	//}

	// 3��������ʧ�����Լ��Ż���
	torch::optim::SGD optimizer(m_net->parameters(), torch::optim::SGDOptions(m_learn_rate[0]));
	torch::nn::CrossEntropyLoss loss_function;  
	loss_function->to(device_type);

	// 4����ʼѵ�� ��ѧϰ�����ŵ������Ӷ���С��
	for (int now_iter = 0; now_iter < Iter; now_iter++)
	{
		if (now_iter == 1)  // �ڵ�2�ε���ʱѧϰ������Ϊm_learn_rate[1]
		{
			updata_learn_rate(optimizer, m_learn_rate[1]);
		}
		if (now_iter == 2)
		{
			updata_learn_rate(optimizer, m_learn_rate[2]);
		}
		// ѵ��
		m_net->train();  
		now_epoch = 0;
		total_loss = 0.0f;
		for (auto& batch : *train_dataLoader)  // �������ݼ�
		{
			now_epoch += 1;
			data = batch.data.to(device_type);  // ͼ�����
			target = batch.target.squeeze().to(device_type);  // ��ǩ
			// ���������Բ鿴ͼ��ߴ�  batch_size * channal * width * height
			//c10::IntArrayRef tsize = data.sizes();
			//int a = tsize[0];
			//int b = tsize[1];
			//int c = tsize[2];
			//int d = tsize[3];
			//std::cout << a << b << c << d << std::endl;

			// ǰ�򴫲�
			prediction = m_net->forward(data);
			// ������ʧ��С
			loss = loss_function(prediction, target);
			total_loss += loss.item<float>();
			// ���ݶȹ����������ݶ��½�
			optimizer.zero_grad(); 
			// ���򴫲� �����ݶ�
			loss.backward();
			// �����ݶȸ���ģ�Ͳ���
			optimizer.step();
			// ��ӡѵ����Ϣ
			if (now_epoch % 5 == 0)
			{
				printf("Iter [%d/%d], Epoch [%d/%d], now_Loss:%.4f Loss: %.4f\n",now_iter,Iter,now_epoch, num_train.value()/batch_size, loss.item<float>() ,total_loss / (now_epoch + 1));
				//std::cout << "Epoch" << i << " Loss=" << total_loss / (i + 1) << std::endl;
			}

		}
		//����
		validation_loss = 0.0f;
		m_net->eval();
		for (auto& batch : *test_dataLoader)
		{
			data = batch.data.to(device_type);  // ͼ����� + ����ΪCUDA
			target = batch.target.squeeze().to(device_type);  // ��ǩ + CUDA
			prediction = m_net->forward(data);
			loss = loss_function(prediction, target);
			validation_loss += loss.item<float>();
		}
		validation_loss /= num_test.value() / batch_size;  // ����ƽ��loss
		(m_best_test_loss > validation_loss) && (m_best_test_loss = validation_loss); // if���ļ��÷�
		printf("get best test loss %.5f", m_best_test_loss);
		torch::save(m_net, "ResNet101_CPP.pt");
	}
}

// ʹ��Pythonת��������ģ���ļ�����Ԥ��
void Result::pred()
{
	torch::jit::script::Module m_net = torch::jit::load("F:/CCCCCProject/AlexNet/Project1/ResNet101.pt", device_type); // ģ�Ͳ�����·������ʵ������޸�
	cv::Mat img = cv::imread(img_root);  // opencv��ȡͼƬ
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);   // BGR��>RGB
	cv::resize(img, img, cv::Size(224, 224));
	// ��opencv������ͼƬת��Tensor���ҽ�BGR��ʽת��RGB��ʽ
	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width   
	img_tensor = torch::unsqueeze(img_tensor,0);
	img_tensor = img_tensor.to(device_type).div(255.0);

	//  ��ʼԤ��
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);
	m_net.eval();
	auto o = m_net.forward(std::move(inputs));
	at::Tensor result = o.toTensor();

	// �õ�Ԥ��Ľ�� result��size = 1 * 10 ������
	std::cout << "��������Ľ����" << result << std::endl;
	auto class1 = torch::max(result,1);
	// ��ӡԤ������
	std::cout << "Ԥ��Ľ���ǣ�" << CLASS_NAME[std::get<1>(class1).item<int>()] << std::endl;
}

// ʹ��C++ѵ���õ���ģ���ļ�����Ԥ��
void Result::pred1()
{
	ResNet m_net(NUM_CLASS);  //NUM_CLASSΪ���ݼ��������
	m_net->to(device_type);
	torch::load(m_net, "ResNet101.pt");  // ģ�Ͳ�����·������ʵ������޸�
	cv::Mat img = cv::imread(img_root);  // opencv��ȡͼƬ
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);   // BGR��>RGB
	cv::resize(img, img, cv::Size(224, 224));
	// ��opencv������ͼƬת��Tensor���ҽ�BGR��ʽת��RGB��ʽ
	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width   
	img_tensor = torch::unsqueeze(img_tensor, 0);
	img_tensor = img_tensor.to(device_type).div(255.0);
	prediction = m_net->forward(img_tensor);
	std::cout << "��������Ľ����" << prediction << std::endl;
	auto class1 = torch::max(prediction, 1);
	// ��ӡԤ������
	std::cout << "Ԥ��Ľ���ǣ�" << CLASS_NAME[std::get<1>(class1).item<int>()] << std::endl;
}