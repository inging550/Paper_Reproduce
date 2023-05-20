#include <iostream>
#include "dataset.h"
#include "result.h"
//#include <torch/csrc/autograd/profiler.h>


int main()
{
	torch::set_num_interop_threads(4);
	Result result;
	// 训练还是预测，取消注释即可运行
	result.train();   
	//result.pred();
	//result.pred1();
}

