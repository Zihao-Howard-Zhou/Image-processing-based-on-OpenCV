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
	imshow("ԭͼ", scr);

	Mat dst;
	dst = MSR(scr, dst);
	imshow("dst", dst);
	waitKey();
	return 0;
}

//-------------------Function 1: Multi-Scale Retinex-------------------//
Mat MSR(Mat& input, Mat& dst)
{
	/*����˵��:
	����MSR����:
	1. ��ԭͼת���������ռ䣬�õ�Log[S(x,y)]
	2. ��ԭͼ����������ͬ�ĳ߶���Gaussian filter����ת�����򣬷ֱ�õ�: Log[L_i(x,y)]
	3. ���� w_i(Log[S(x,y)] - Log[L_i(x,y)])������������ӣ��õ�Log[R(x,y)]
	4. ���ڴ�ʱ�� Log[R(x,y)] ��ֵ������0-255��Χ�ڣ�������Ҫ������������
	5. ���ת����ʽ���õ� r(x,y) ���
	*/

	int gain = 128;
    int offset = 128;
	Mat input_f, input_log, Li_log;           //��ΪLog[S(x,y)]

	vector<double> sigemas;

	sigemas.push_back(30);
	sigemas.push_back(150);
	sigemas.push_back(300);

	//��ʼ��Ȩ��
	vector<double> weights;
	for (int i = 0; i < 3; i++)
    {
       weights.push_back(1.f / 3);           //Ȩ��Ϊ1/3
    }

	input.convertTo(input_f, CV_32FC3);     
	log(input_f, input_log);                 //�õ��������µ� Log[S(x,y)]


    //���ݸ�����Ȩ�ع�һ��
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

	
	vector<Mat> log_term;                    //����������

	for(size_t i = 0; i < num; i++)
	{
		Mat blur = input_f.clone();
		gaussianFilter(blur, sigemas[i]);    //������˹�˲�֮��õ� L_i(x,y)
		log(blur, Li_log);                   //�൱��log[L_i(x,y)]
		log_term.push_back(weights[i]*(input_log - Li_log));    //��һ��ִ�о��ǵ�0���������ĵ�һλ  w_i(Log[S(x,y)] - log[L_i(x,y)])
	}

	Mat dst2;
	dst = log_term[0] + log_term[1] + log_term[2];

	
	dst = (dst2 * gain) + offset;            //ʱ��dstֵ�ķ�Χ������0�C255�����Ի���Ҫ�����������첢ת������Ӧ�ĸ�ʽ�����ʾ
	dst.convertTo(dst, CV_8UC3);             //ת��������������ʾ��8UC3��ʽ
	return dst;
}

//-----------Function2: ��˹ģ��---------------//
void gaussianFilter(Mat &img, double sigma)
{
	/*����˵��:
	���Function ʵ�ֵ��Ǹ�˹ģ��, ������Ҫ�Ĵ��벿�����ڼ���kernel size
	���������� kernel size �Ĺ�ʽ��: 6 * sigma +1                 ��1��
	���ڹ�ʽ��1���Ľ���: ������̬�ֲ��� 3�� ׼��, ���Ա���������3�ҵķ�Χ֮��, ��˹�����ĸ����ܶ�ֵ����Ϊ0
	���, ��Ч�Ĵ��ڴ�СӦΪ 6��, ������ GaussianBlur ��������Ҫ�� kernel sizeΪ����, �������+1����
	*/

    int filter_size;
    
    if (sigma > 300)
    {
        sigma = 300;                //���ܴ���300������300ǿ�иĳ�300
    }

    //��ȡ�˲����Ĵ�С��תΪ����
    filter_size = (int)floor(sigma * 6) +1;
	
    //���С��3�򷵻�
    if (filter_size < 3)
    {
        return;
    }

    GaussianBlur(img, img, Size(filter_size, filter_size), 0);    //���и�˹ģ��
}
