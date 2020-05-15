//---------------------------Dark Channel Prior-----------------------//
//Configuration: Visual Studio 2010 + Opencv2.4.10
//Author: ZiHao Zhou / South China University of Technology 
//Date: 2020.5.15

#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
using namespace std;
using namespace cv;

void gaussianFilter(Mat &img, double sigma);
Mat MSR(Mat& input, Mat& dst);
Mat MSRCR(Mat& input, Mat& MSR_input, double alpha, double beta, double dynamic, int gain, int offset, Mat& output);

int main()
{
	Mat scr;
	scr = imread("test6.jpg", 1);
	imshow("原图", scr);
	Mat log_R, dst;
	log_R= MSR(scr, dst);
	
	Mat dst1, dst2, output;
	output = MSRCR(scr, log_R, 128.0, 1.0, 3.0, 128, 128, output);
	imshow("MSRCR效果图", output);

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

	sigemas.push_back(10);
	sigemas.push_back(100);
	sigemas.push_back(200);

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
	dst2 = log_term[0] + log_term[1] + log_term[2];

	
	dst = (dst2 * gain) + offset;            //时的dst值的范围并不是0–255，所以还需要进行线性拉伸并转换成相应的格式输出显示
	dst.convertTo(dst, CV_8UC3);             //转换成正常用于显示的8UC3格式
	return dst2;   //dst2就是Log[R(x,y)]
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
	cout<<filter_size<<" ";
	
    //如果小于3则返回
    if (filter_size < 3)
    {
        return;
    }

    GaussianBlur(img, img, Size(filter_size, filter_size), 0);    //进行高斯模糊
}

//---------------------------------------------Function MSRCR_CS------------------------------------------------------------//
Mat MSRCR(Mat& input, Mat& log_MSR_input, double alpha, double beta, double dynamic, int gain, int offset, Mat& output)
{
	/*函数说明:
	参数1:  输入的原始图像
	参数2: MSR算法所得到的对数域下的图像, 也就是 log[R_{MSR}(x,y)]
	参数3: alpha参数
	参数4: beta参数
	参数5: dynamic参数: 动态参数（dynamic越小，图像的对比度就越强，此处选择dynamic=1效果和GIMP软件一致)
	参数6: 增益gain
	参数7: 偏置offset
	参数8: MSRCR算法运行的输出图像
	*/

	Mat output2;
	output2 = input.clone();
	output2.convertTo(output2, CV_32FC3);

	output = input.clone();
	output.convertTo(output, CV_32FC3);
	vector<Mat> split_input, split_MSR_input,split_output;
	split(input, split_input);                                            //把I(x,y)分解，type:CV_8UC1
	split(output, split_output);

	//cout<<"sss"<<split_output[0].type();

	Mat add = split_input[0] + split_input[1] + split_input[2];           //求出 sum_{i=1}^3I_i(x,y)
	add.convertTo(add, CV_32FC1);                                         //必须要转化成CV_32F的类型，否则无法进行乘法、除法的运算

	Mat A1, A2, A3;
	A1 = split_input[0];
	A2 = split_input[1];
	A3 = split_input[2];

	A1.convertTo(A1, CV_32FC1);                                           //Mat的乘法需要的类型是 CV_32FC类型
	A2.convertTo(A2, CV_32FC1);
	A3.convertTo(A3, CV_32FC1);

	A1 = A1.mul(alpha);                                                   //αI_i(x,y)
	A2 = A2.mul(alpha);
	A3 = A3.mul(alpha);

	A1 = A1 / add;
	A2 = A2 / add;
	A3 = A3 / add;


	log(A1, A1);
	log(A2, A2);
	log(A3, A3);

	A1 = A1.mul(beta);                                                    //得到了C_i
	A2 = A2.mul(beta);
	A3 = A3.mul(beta);
	
	split(log_MSR_input, split_MSR_input);                                //我们需要将C_i和 R_{MSR_i}(x,y)相乘，而这里的R_{MSR_i}(x,y)的形式应该是Log[R(x,y)]

	cout<<log_MSR_input.type()<<endl;
	
	cout<<split_MSR_input[0].rows<<" "<<split_MSR_input[0].cols<<endl;
	cout<<A1.rows<<" "<<A1.cols<<endl;

	split_MSR_input[0].convertTo(split_MSR_input[0], CV_32FC1);   
	split_MSR_input[1].convertTo(split_MSR_input[1], CV_32FC1);
	split_MSR_input[2].convertTo(split_MSR_input[2], CV_32FC1);


	split_MSR_input[0] = split_MSR_input[0].mul(A1);                       //这个就是 log[R_{MSRCR_i}(x,y)]
	split_MSR_input[1] = split_MSR_input[1].mul(A2);
	split_MSR_input[2] = split_MSR_input[2].mul(A3);
	

	merge(split_MSR_input, output2);                                        //得到了 R_MSRCR(x,y)

	//output = (output * gain) + offset;                                    //时的dst值的范围并不是0–255，所以还需要进行线性拉伸并转换成相应的格式输出显示
	//output.convertTo(output, CV_8UC3);                                    //转换成正常用于显示的8UC3格式

	
	Mat means, stddev;
	meanStdDev(output2, means, stddev);                                     //计算Log[R(x,y)]的均值和方差

	
	double min1, max1, min2, max2, min3, max3;
	min1 = means.at<double>(0) - dynamic * stddev.at<double>(0);            //第一个通道的min和max, Dynamic = 1
	max1 = means.at<double>(0) + dynamic * stddev.at<double>(0);  

	min2 = means.at<double>(1) - dynamic * stddev.at<double>(1);  
	max2 = means.at<double>(1) + dynamic * stddev.at<double>(1);  

	min3 = means.at<double>(2) - dynamic * stddev.at<double>(2);  
	max3 = means.at<double>(2) + dynamic * stddev.at<double>(2);  


	for(int rows = 0; rows < split_MSR_input[0].rows; rows++)
	{
		for(int cols = 0; cols < split_MSR_input[0].cols; cols++)
		{
			split_output[0].at<float>(rows, cols) = (split_MSR_input[0].at<float>(rows, cols) - min1) / (max1 - min1) * (255 - 0);
			
			if(split_output[0].at<float>(rows, cols) > 255)                  //溢出控制
			{
				split_output[0].at<float>(rows, cols) = 255;
			}
			else
			{
				if(split_output[0].at<float>(rows, cols) < 0)
				{
					split_output[0].at<float>(rows, cols) = 0;
				}
			}
			split_output[1].at<float>(rows, cols) = (split_MSR_input[1].at<float>(rows, cols) - min1) / (max1 - min1) * (255 - 0);

			if(split_output[1].at<float>(rows, cols) > 255)
			{
				split_output[1].at<float>(rows, cols) = 255;
			}
			else
			{
				if(split_output[1].at<float>(rows, cols) < 0)
				{
					split_output[1].at<float>(rows, cols) = 0;
				}
			}
			split_output[2].at<float>(rows, cols) = (split_MSR_input[2].at<float>(rows, cols) - min1) / (max1 - min1) * (255 - 0);

			if(split_output[2].at<float>(rows, cols) > 255)
			{
				split_output[2].at<float>(rows, cols) = 255;
			}
			else
			{
				if(split_output[2].at<float>(rows, cols) < 0)
				{
					split_output[2].at<float>(rows, cols) = 0;
				}
			}
			
		}
	}

	merge(split_output, output);
	output.convertTo(output, CV_8UC3);
	
	return output;
}

