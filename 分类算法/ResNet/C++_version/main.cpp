#include <iostream>
#include "dataset.h"
#include "result.h"
//#include <torch/csrc/autograd/profiler.h>


int main()
{
	torch::set_num_interop_threads(4);
	Result result;
	// ѵ������Ԥ�⣬ȡ��ע�ͼ�������
	result.train();   
	//result.pred();
	//result.pred1();
}

