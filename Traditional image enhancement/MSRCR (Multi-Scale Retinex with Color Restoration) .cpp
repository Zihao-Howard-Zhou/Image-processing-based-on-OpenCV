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
	imshow("ԭͼ", scr);
	Mat log_R, dst;
	log_R= MSR(scr, dst);
	
	Mat dst1, dst2, output;
	output = MSRCR(scr, log_R, 128.0, 1.0, 3.0, 128, 128, output);
	imshow("MSRCRЧ��ͼ", output);

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

	sigemas.push_back(10);
	sigemas.push_back(100);
	sigemas.push_back(200);

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
	dst2 = log_term[0] + log_term[1] + log_term[2];

	
	dst = (dst2 * gain) + offset;            //ʱ��dstֵ�ķ�Χ������0�C255�����Ի���Ҫ�����������첢ת������Ӧ�ĸ�ʽ�����ʾ
	dst.convertTo(dst, CV_8UC3);             //ת��������������ʾ��8UC3��ʽ
	return dst2;   //dst2����Log[R(x,y)]
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
	cout<<filter_size<<" ";
	
    //���С��3�򷵻�
    if (filter_size < 3)
    {
        return;
    }

    GaussianBlur(img, img, Size(filter_size, filter_size), 0);    //���и�˹ģ��
}

//---------------------------------------------Function MSRCR_CS------------------------------------------------------------//
Mat MSRCR(Mat& input, Mat& log_MSR_input, double alpha, double beta, double dynamic, int gain, int offset, Mat& output)
{
	/*����˵��:
	����1:  �����ԭʼͼ��
	����2: MSR�㷨���õ��Ķ������µ�ͼ��, Ҳ���� log[R_{MSR}(x,y)]
	����3: alpha����
	����4: beta����
	����5: dynamic����: ��̬������dynamicԽС��ͼ��ĶԱȶȾ�Խǿ���˴�ѡ��dynamic=1Ч����GIMP���һ��)
	����6: ����gain
	����7: ƫ��offset
	����8: MSRCR�㷨���е����ͼ��
	*/

	Mat output2;
	output2 = input.clone();
	output2.convertTo(output2, CV_32FC3);

	output = input.clone();
	output.convertTo(output, CV_32FC3);
	vector<Mat> split_input, split_MSR_input,split_output;
	split(input, split_input);                                            //��I(x,y)�ֽ⣬type:CV_8UC1
	split(output, split_output);

	//cout<<"sss"<<split_output[0].type();

	Mat add = split_input[0] + split_input[1] + split_input[2];           //��� sum_{i=1}^3I_i(x,y)
	add.convertTo(add, CV_32FC1);                                         //����Ҫת����CV_32F�����ͣ������޷����г˷�������������

	Mat A1, A2, A3;
	A1 = split_input[0];
	A2 = split_input[1];
	A3 = split_input[2];

	A1.convertTo(A1, CV_32FC1);                                           //Mat�ĳ˷���Ҫ�������� CV_32FC����
	A2.convertTo(A2, CV_32FC1);
	A3.convertTo(A3, CV_32FC1);

	A1 = A1.mul(alpha);                                                   //��I_i(x,y)
	A2 = A2.mul(alpha);
	A3 = A3.mul(alpha);

	A1 = A1 / add;
	A2 = A2 / add;
	A3 = A3 / add;


	log(A1, A1);
	log(A2, A2);
	log(A3, A3);

	A1 = A1.mul(beta);                                                    //�õ���C_i
	A2 = A2.mul(beta);
	A3 = A3.mul(beta);
	
	split(log_MSR_input, split_MSR_input);                                //������Ҫ��C_i�� R_{MSR_i}(x,y)��ˣ��������R_{MSR_i}(x,y)����ʽӦ����Log[R(x,y)]

	cout<<log_MSR_input.type()<<endl;
	
	cout<<split_MSR_input[0].rows<<" "<<split_MSR_input[0].cols<<endl;
	cout<<A1.rows<<" "<<A1.cols<<endl;

	split_MSR_input[0].convertTo(split_MSR_input[0], CV_32FC1);   
	split_MSR_input[1].convertTo(split_MSR_input[1], CV_32FC1);
	split_MSR_input[2].convertTo(split_MSR_input[2], CV_32FC1);


	split_MSR_input[0] = split_MSR_input[0].mul(A1);                       //������� log[R_{MSRCR_i}(x,y)]
	split_MSR_input[1] = split_MSR_input[1].mul(A2);
	split_MSR_input[2] = split_MSR_input[2].mul(A3);
	

	merge(split_MSR_input, output2);                                        //�õ��� R_MSRCR(x,y)

	//output = (output * gain) + offset;                                    //ʱ��dstֵ�ķ�Χ������0�C255�����Ի���Ҫ�����������첢ת������Ӧ�ĸ�ʽ�����ʾ
	//output.convertTo(output, CV_8UC3);                                    //ת��������������ʾ��8UC3��ʽ

	
	Mat means, stddev;
	meanStdDev(output2, means, stddev);                                     //����Log[R(x,y)]�ľ�ֵ�ͷ���

	
	double min1, max1, min2, max2, min3, max3;
	min1 = means.at<double>(0) - dynamic * stddev.at<double>(0);            //��һ��ͨ����min��max, Dynamic = 1
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
			
			if(split_output[0].at<float>(rows, cols) > 255)                  //�������
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

