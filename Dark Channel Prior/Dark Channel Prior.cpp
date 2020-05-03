//---------------------------Dark Channel Prior-----------------------//
//Configuration: Visual Studio 2010 + Opencv2.4.10
//Author: ZiHao Zhou / South China University of Technology 
//Date: 2020.5.3

#include <opencv2/core/core.hpp>      
#include <opencv2/highgui/highgui.hpp>      
#include <opencv2/imgproc/imgproc.hpp>     
#include <iostream>  
#include<vector>

using namespace std;
using namespace cv;

Mat darkchannel(Mat& inputImage, Mat& outputImage, int r);                            //��ȡͼ��ͨ��
double getGlobelAtmosphericLight(const Mat darkChannelImg);                           //���������ֵ
Mat getTransimissionImg(const Mat darkChannelImg,const double A);
Mat GuideFilter(Mat& I, Mat& P, int radius, double eps);
Mat getDehazedImg_guidedFilter(Mat src,Mat darkChannelImg);
Mat GuidedFilter(Mat I_org, Mat p_org, int r, double eps, int s);



int main()
{
	Mat src, darkChanelImg;
	src = imread("test1.jpg", 1);
	imshow("ԭͼ��", src);

	src.convertTo(src, CV_8U, 1, 0);

	darkChanelImg = darkchannel(src,darkChanelImg, 15);
	imshow("ͼ��İ�ͨ��", darkChanelImg);

	double A = getGlobelAtmosphericLight(darkChanelImg);

	Mat transmissionImage(darkChanelImg.size(), darkChanelImg.type());
	transmissionImage = getTransimissionImg(darkChanelImg, A);
	imshow("���Դ���ͼ", transmissionImage);

	Mat dehazedImg, dehazedImg_guideFilter;
	double t = (double)getTickCount();
	dehazedImg_guideFilter = getDehazedImg_guidedFilter(src, darkChanelImg);
	t = (double)getTickCount() - t;
	imshow("ȥ��ͼ", dehazedImg_guideFilter);
	cout <<"ͼ��ȥ������ʱ��Ϊ:"<<1000 * t / (getTickFrequency()) << "ms" << std::endl;

	imwrite("rescult.jpg", dehazedImg_guideFilter);

	waitKey();
	return 0;
}


//-------------------------��Function 1��: Get dark channel-------------------------//
Mat darkchannel(Mat& inputImage, Mat& outputImage, int r)
{
	/*
	����˵��:
	1. ��������˵��: ����������õ�������ͼ��İ�ͨ��
	2. ����������˵��
	   ����1: ����ͼ��
	   ����2: ����ͼ��
	   ����3: ����Сֵ�˲���filter�ߴ�
	3. ����������ͼ�����صİ취���ڶ�̬��ַ����(���ǱȽ���������һ�ְ취,����Ϊ�����Ч�ʿ��Բ���
	   isContinuous()����������ÿһ������
	*/

	Mat element;
	vector<Mat> channel_of_input;                      
	split(inputImage, channel_of_input);
	int rows = inputImage.rows;
	int cols = inputImage.cols;
	outputImage = channel_of_input.at(0).clone();      //����һ�ź͵�ͨ��ͼ��Сһ��

	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < cols; j++)
		{
			int x = channel_of_input.at(0).at<uchar>(i, j);
			int y = channel_of_input.at(1).at<uchar>(i, j);
			int z = channel_of_input.at(2).at<uchar>(i,j);
			outputImage.at<uchar>(i, j) = x < y ? (x < z ? x : z) : (y < z ? y : z);
		}
	}
	element = getStructuringElement(MORPH_RECT, Size(r, r));
	erode(outputImage,outputImage, element, Point(-1,-1), 1,0);        //erode��ʴ����ǡ�þ��Ǽ���ֲ���Сֵ

	return outputImage;
}

//-------------------------��Function2��getGlobelAtmosphericLight()����-------------------------//
double getGlobelAtmosphericLight(const Mat darkChannelImg)
{
	/*
	��������˵��:���������ڼ���A
	�����Ǽ򻯵Ĵ���ʽ,A�����ֵ�޶�Ϊ220
	*/

	double minAtomsLight = 220;     //����ֵ
	double maxValue = 0;
	Point maxLoc;
	minMaxLoc(darkChannelImg, NULL, &maxValue, NULL, &maxLoc);
	double A = min(minAtomsLight,maxValue);
	return A;
}

//---------------��Function3��getTransimissionImg()����--------------------------//
Mat getTransimissionImg(const Mat darkChannelImg,const double A)
{
	Mat transmissionImg(darkChannelImg.size(),CV_8UC1);
	Mat look_up(1,256,CV_8UC1);
 
	uchar* look_up_ptr = look_up.data; 
	for (int k = 0; k < 256; k++)
	{
		look_up_ptr[k] = cv::saturate_cast<uchar>(255*(1 - 0.95 * k / A));
	}
 
	LUT(darkChannelImg, look_up, transmissionImg);
 
	return transmissionImg;
}

//-------------------------��Function4�� Guidefilter����-------------------------//
Mat GuidedFilter(Mat I_org, Mat p_org, int r, double eps, int s)
{
	/*
	��Ҫ����˵��:
	����1: I_org ��ʾ����ͼ,��ΪscrMat����
	����2: p_org ��ʾ���˲�ͼ��
	����3: �˲��뾶
	˵��:������ʵ�ֵ��ǶԵ�һͨ�����еĵ����˲�
	*/
 
	Mat I, _I;
	I_org.convertTo(_I, CV_64FC1, 1.0 / 255);
	resize(_I, I, Size(), 1.0 / s, 1.0 / s, 1);
 
	Mat p, _p;
	p_org.convertTo(_p, CV_64FC1, 1.0 / 255);
	//p = _p;
	resize(_p, p, Size(), 1.0 / s, 1.0 / s, 1);
    
	int hei = I.rows;
	int wid = I.cols;
 
	r = (2 * r + 1) / s + 1;                                    //opencv��boxFilter()�е�Size,����9x9,���Ǿ�˵�뾶Ϊ4   
 
	Mat mean_I;
	boxFilter(I, mean_I, CV_64FC1, Size(r, r));                 //������� I ͼ�ľ�ֵ,Ҳ���Ǧ�_k
 
	Mat mean_p;
	boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));             //������� P ͼ�ľ�ֵ��Ҳ���� p_k
  
	Mat mean_Ip;
	boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));     //������� I��P �ľ�ֵ,a_k��ĸ�ĵ�һ��
   
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);                  //����� ak �ķ���
	 
	Mat mean_II;
	boxFilter(I.mul(I), mean_II, CV_64FC1, Size(r, r));          //������� I��I �ľ�ֵ, ��Ϊ���㷽��� EX^2
   
	Mat var_I = mean_II - mean_I.mul(mean_I);                    //�õ�I�ķ���
       
	Mat a = cov_Ip / (var_I + eps);
	Mat b = mean_p - a.mul(mean_I);

	Mat mean_a;
	boxFilter(a, mean_a, CV_64FC1, Size(r, r));                  //���� ak �ľ�ֵ
	Mat rmean_a;
	resize(mean_a, rmean_a, Size(I_org.cols, I_org.rows), 1);
   
	Mat mean_b;
	boxFilter(b, mean_b, CV_64FC1, Size(r, r));                   //���� bk �ľ�ֵ
	Mat rmean_b;
	resize(mean_b, rmean_b, Size(I_org.cols, I_org.rows), 1);
    
	Mat q = rmean_a.mul(_I) + rmean_b;
	Mat q1;
	q.convertTo(q1, CV_8UC1, 255, 0);
 
	return q1;
}

//-----------------------��funtion 6��getDehazedChanne-------------------------------//
Mat getDehazedChannel(Mat srcChannel, Mat transmissionChannel, double A)
{
	/*����˵��:
	��������: �����������ڼ��㵥��ͨ����ȥ��ͼ��
	����1: ����ԭͼ���һ��ͨ��
	����2: ����ͼ�ĵ�һͨ��(��Ҫ��ԭͼ��ͨ����Ӧ)
	����3: ȫ�������ֵA
	*/

	double tmin = 0.1;
	double tmax;
 
	Mat dehazedChannel(srcChannel.size(), CV_8UC1);
	for (int i = 0; i<srcChannel.rows; i++)
	{
		for (int j = 0; j<srcChannel.cols; j++)
		{
			double transmission = transmissionChannel.at<uchar>(i, j);
 
			tmax = (transmission / 255) < tmin ? tmin : (transmission / 255);
			//(I-A)/t +A  
			dehazedChannel.at<uchar>(i, j) = saturate_cast<uchar>(abs((srcChannel.at<uchar>(i, j) - A) / tmax + A));
		}
	}
	return dehazedChannel;
}

//-----------------��Function 7��getDehazedImg_guidedFilter-----------------------//
Mat getDehazedImg_guidedFilter(Mat src,Mat darkChannelImg)
{
	/*����˵��:
	�������������ռ���õ�ȥ��Ч��ͼ
	����1: �����3ͨ��ͼ��
	����2: ����İ�ͨ��ͼ��
	*/

	Mat dehazedImg = Mat::zeros(src.rows, src.cols, CV_8UC3);             //����һ��3ͨ������ȥ��ͼ��
 
	Mat transmissionImg(src.rows, src.cols, CV_8UC3);
	Mat fineTransmissionImg(src.rows, src.cols, CV_8UC3);
	vector<Mat> srcChannel, dehazedChannel, transmissionChannel, fineTransmissionChannel;
 
	split(src, srcChannel);
	double A0 = getGlobelAtmosphericLight(darkChannelImg);
	double A1 = getGlobelAtmosphericLight(darkChannelImg);
	double A2 = getGlobelAtmosphericLight(darkChannelImg);
 
	split(transmissionImg, transmissionChannel);
	transmissionChannel.at(0) = getTransimissionImg(darkChannelImg, A0);                                    //�õ�ÿһ��ͨ�����ԵĴ���ͼ
	transmissionChannel.at(1) = getTransimissionImg(darkChannelImg, A1);
	transmissionChannel.at(2) = getTransimissionImg(darkChannelImg, A2);
 
	split(fineTransmissionImg, fineTransmissionChannel);
	fineTransmissionChannel.at(0) = GuidedFilter(srcChannel.at(0), transmissionChannel.at(0), 64, 0.01, 8);  //���������˲��õ���ϸ����ûһ��ͨ���Ĵ���ͼ
	fineTransmissionChannel.at(1) = GuidedFilter(srcChannel.at(1), transmissionChannel.at(1), 64, 0.01, 8);
	fineTransmissionChannel.at(2) = GuidedFilter(srcChannel.at(2), transmissionChannel.at(2), 64, 0.01, 8);
 
	merge(fineTransmissionChannel, fineTransmissionImg);
	imshow("fineTransmissionChannel", fineTransmissionImg);
	
	split(dehazedImg, dehazedChannel);
	dehazedChannel.at(0) = getDehazedChannel(srcChannel.at(0), fineTransmissionChannel.at(0), A0);
	dehazedChannel.at(1) = getDehazedChannel(srcChannel.at(1), fineTransmissionChannel.at(1), A1);
	dehazedChannel.at(2) = getDehazedChannel(srcChannel.at(2), fineTransmissionChannel.at(2), A2);
 
	merge(dehazedChannel, dehazedImg);
 
	return dehazedImg;
}
 
