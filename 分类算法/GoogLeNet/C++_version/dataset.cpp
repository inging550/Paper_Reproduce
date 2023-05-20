# include "dataset.h"

void dataSetClc::load_data_from_folder(std::string path, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int label)
{
	// ��������
	long long hFile = 0; //���
	struct _finddata_t fileInfo;  // _finddata_tΪһ���ṹ��
	std::string pathName;

	if ((hFile = _findfirst(pathName.assign(path).append("\\*.*").c_str(), &fileInfo)) == -1)
	{
		return;
	}
	do
	{
		const char* s = fileInfo.name;
		const char* t = type.data();

		if (fileInfo.attrib&_A_SUBDIR) //�����ļ���
		{
			//�������ļ����е��ļ�(��)
			if (strcmp(s, ".") == 0 || strcmp(s, "..") == 0) //���ļ���Ŀ¼��.����..
				continue;
			std::string sub_path = path + "\\" + fileInfo.name;
			label++;
			load_data_from_folder(sub_path, type, list_images, list_labels, label);

		}
		else //�ж��ǲ��Ǻ�׺Ϊtype�ļ�
		{
			if (strstr(s, t))
			{
				std::string image_path = path + "\\" + fileInfo.name;
				// ��ͼ��·���Լ���Ӧ��ǩ���vector������
				list_images.push_back(image_path);
				list_labels.push_back(label);
			}
		}
	} while (_findnext(hFile, &fileInfo) == 0);
}

torch::data::Example<> dataSetClc::get(size_t index)
{
	std::string image_path = image_paths.at(index);  //vector����Ƭ
	cv::Mat image = cv::imread(image_path);   // opencv��ȡͼ��
	cv::resize(image, image, cv::Size(224, 224)); //�ߴ�ͳһ
	cv::cvtColor(image,image,cv::COLOR_BGR2RGB);   // BGR��>RGB
	int label = labels.at(index);   // ��ȡ�����Ϣ
	// ��opencv��ʽ�ľ���ת��Ϊ����
	torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width
	torch::Tensor label_tensor = torch::full({ 1 }, label);
	return { img_tensor, label_tensor };
}

torch::optional<size_t> dataSetClc::size() const
{
	return image_paths.size();
};