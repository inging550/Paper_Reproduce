#pragma once
# include "dataset.h"
#include <opencv2/core/core.hpp>
#include <torch/script.h>

class Result
{
private:
	torch::DeviceType device_type;
	int batch_size;  // ѵ����batch_size
	int NUM_CLASS;  // �������
	const char* img_root;  // ��Ԥ��ͼƬ·��
	int Iter = 3;   // ѵ�����ܵ�������
	float m_best_test_loss = 20.0f;
	float m_learn_rate[3] = { 1E-2,1E-4,1E-5 };
	const char* CLASS_NAME[10] = { "����0","����1","����2","����3","����4","����5","����6","����7","����8","����9" };  // �����Լ����ݼ����и��ģ���д˳����ѵ�������ļ����������϶��µ�˳��
public:
	Result();
	void train();
	void pred();
	void pred1();
	void updata_learn_rate(torch::optim::SGD &optimizer, double alpha);
private:
	// ����ѵ��ʱ��һЩ����
	int now_epoch;
	float total_loss;
	at::Tensor data;  // ͼ�����
	at::Tensor target;  // ��ǩֵ
	torch::Tensor prediction;  // ���������ֵ
	torch::Tensor loss; // ��ʧֵ
	// ����ʱ��һЩ����
	float validation_loss;
};
