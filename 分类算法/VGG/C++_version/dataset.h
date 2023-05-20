#pragma once
# include "model.h"
# include <opencv2/opencv.hpp>
# include <fstream>
# include <string>
#include <io.h>


class dataSetClc : public torch::data::Dataset<dataSetClc> 
{
public:
	int class_index = 0;
	// �˺���Ŀ��Ϊ��ȡ���е�ͼƬ�Լ���Ӧ��ǩ���洢��list_images�Լ�list_labels��
	// typeΪ��׺��
	void load_data_from_folder(std::string path, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int label, int data_num);	
	// ���캯��
	dataSetClc(std::string image_dir, std::string type, int data_num) {
		//image_paths.reserve();
		load_data_from_folder(image_dir, std::string(type), image_paths, labels, class_index - 1, data_num);
	}
	// ��дget()����������index��Ӧ��ͼƬ�����Լ���ǩ
	torch::data::Example<> get(size_t index) override;
	// ��дsize()�������������ݼ�����
	torch::optional<size_t> size() const override;
private:
	std::vector<std::string> image_paths;  // �����ļ��е�˳��洢���ݼ������е�ͼƬ·��
	std::vector<int> labels;   // �����ļ��е�˳��洢ͼƬ·����Ӧ������ǩ
};