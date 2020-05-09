//--------------------------- MSR Multi-Scale Retinex-----------------------//
//Configuration: Visual Studio 2010 + Opencv2.4.10
//Author: ZiHao Zhou / South China University of Technology 
//Date: 2020.5.9


#include <iostream>    
#include <opencv2\opencv.hpp>    
#include <opencv2\highgui\highgui.hpp>    
using namespace std;
using namespace cv;


void gaussianFilter(Mat &img, double sigma);
Mat MSR(Mat& input, Mat& dst);

int main()
{
	Mat scr = imread("test2.jpg", 1);
	imshow("原图", scr);

	Mat dst;
	dst = MSR(scr, dst);
	imshow("dst", dst);
	waitKey();
	return 0;
}

//-------------------Function 1: Multi-Scale Retinex-------------------//
Mat MSR(Mat& input, Mat& dst)
{
	/*函数说明:
	基本MSR步骤:
	1. 将原图转换到对数空间，得到Log[S(x,y)]
	2. 将原图按照三个不同的尺度做Gaussian filter，再转对数域，分别得到: Log[L_i(x,y)]
	3. 计算 w_i(Log[S(x,y)] - Log[L_i(x,y)])，将三个项相加，得到Log[R(x,y)]
	4. 由于此时的 Log[R(x,y)] 的值并不在0-255范围内，所以需要进行线性拉伸
	5. 最后转换格式，得到 r(x,y) 输出
	*/

	int gain = 128;
    int offset = 128;
	Mat input_f, input_log, Li_log;           //作为Log[S(x,y)]

	vector<double> sigemas;

	sigemas.push_back(30);
	sigemas.push_back(150);
	sigemas.push_back(300);

	//初始化权重
	vector<double> weights;
	for (int i = 0; i < 3; i++)
    {
       weights.push_back(1.f / 3);           //权重为1/3
    }

	input.convertTo(input_f, CV_32FC3);     
	log(input_f, input_log);                 //得到对数域下的 Log[S(x,y)]


    //根据给定的权重归一化
    double weight = 0;
    size_t num = weights.size();
    for (size_t i = 0; i < num; i++)
    {
        weight += weights[i];
    }

    if (weight != 1.0f)
    {
        input_log *= weight;
    }

	
	vector<Mat> log_term;                    //对数项容器

	for(size_t i = 0; i < num; i++)
	{
		Mat blur = input_f.clone();
		gaussianFilter(blur, sigemas[i]);    //经过高斯滤波之后得到 L_i(x,y)
		log(blur, Li_log);                   //相当于log[L_i(x,y)]
		log_term.push_back(weights[i]*(input_log - Li_log));    //第一次执行就是第0项，将在数组的第一位  w_i(Log[S(x,y)] - log[L_i(x,y)])
	}

	Mat dst2;
	dst = log_term[0] + log_term[1] + log_term[2];

	
	dst = (dst2 * gain) + offset;            //时的dst值的范围并不是0–255，所以还需要进行线性拉伸并转换成相应的格式输出显示
	dst.convertTo(dst, CV_8UC3);             //转换成正常用于显示的8UC3格式
	return dst;
}

//-----------Function2: 高斯模糊---------------//
void gaussianFilter(Mat &img, double sigma)
{
	/*函数说明:
	这个Function 实现的是高斯模糊, 但是主要的代码部分在于计算kernel size
	本函数计算 kernel size 的公式是: 6 * sigma +1                 （1）
	关于公式（1）的解释: 根据正态分布的 3σ 准则, 当自变量超过±3σ的范围之后, 高斯函数的概率密度值几乎为0
	因此, 有效的窗口大小应为 6σ, 但由于 GaussianBlur 函数里面要求 kernel size为奇数, 因此做了+1处理
	*/

    int filter_size;
    
    if (sigma > 300)
    {
        sigma = 300;                //不能大于300，大于300强行改成300
    }

    //获取滤波器的大小，转为奇数
    filter_size = (int)floor(sigma * 6) +1;
	
    //如果小于3则返回
    if (filter_size < 3)
    {
        return;
    }

    GaussianBlur(img, img, Size(filter_size, filter_size), 0);    //进行高斯模糊
}
