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

Mat darkchannel(Mat& inputImage, Mat& outputImage, int r);                            //获取图像暗通道
double getGlobelAtmosphericLight(const Mat darkChannelImg);                           //计算大气光值
Mat getTransimissionImg(const Mat darkChannelImg,const double A);
Mat GuideFilter(Mat& I, Mat& P, int radius, double eps);
Mat getDehazedImg_guidedFilter(Mat src,Mat darkChannelImg);
Mat GuidedFilter(Mat I_org, Mat p_org, int r, double eps, int s);



int main()
{
	Mat src, darkChanelImg;
	src = imread("test1.jpg", 1);
	imshow("原图像", src);

	src.convertTo(src, CV_8U, 1, 0);

	darkChanelImg = darkchannel(src,darkChanelImg, 15);
	imshow("图像的暗通道", darkChanelImg);

	double A = getGlobelAtmosphericLight(darkChanelImg);

	Mat transmissionImage(darkChanelImg.size(), darkChanelImg.type());
	transmissionImage = getTransimissionImg(darkChanelImg, A);
	imshow("粗略传输图", transmissionImage);

	Mat dehazedImg, dehazedImg_guideFilter;
	double t = (double)getTickCount();
	dehazedImg_guideFilter = getDehazedImg_guidedFilter(src, darkChanelImg);
	t = (double)getTickCount() - t;
	imshow("去雾图", dehazedImg_guideFilter);
	cout <<"图像去雾所用时间为:"<<1000 * t / (getTickFrequency()) << "ms" << std::endl;

	imwrite("rescult.jpg", dehazedImg_guideFilter);

	waitKey();
	return 0;
}


//-------------------------【Function 1】: Get dark channel-------------------------//
Mat darkchannel(Mat& inputImage, Mat& outputImage, int r)
{
	/*
	函数说明:
	1. 函数功能说明: 本函数计算得到了输入图像的暗通道
	2. 函数参数的说明
	   参数1: 输入图像
	   参数2: 返回图像
	   参数3: 做最小值滤波的filter尺寸
	3. 本函数处理图像像素的办法基于动态地址计算(这是比较容易理解的一种办法,不过为了提高效率可以采用
	   isContinuous()函数来访问每一个像素
	*/

	Mat element;
	vector<Mat> channel_of_input;                      
	split(inputImage, channel_of_input);
	int rows = inputImage.rows;
	int cols = inputImage.cols;
	outputImage = channel_of_input.at(0).clone();      //复制一张和单通道图大小一样

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
	erode(outputImage,outputImage, element, Point(-1,-1), 1,0);        //erode腐蚀操作恰好就是计算局部最小值

	return outputImage;
}

//-------------------------【Function2】getGlobelAtmosphericLight()函数-------------------------//
double getGlobelAtmosphericLight(const Mat darkChannelImg)
{
	/*
	函数功能说明:本函数用于计算A
	这里是简化的处理方式,A的最大值限定为220
	*/

	double minAtomsLight = 220;     //经验值
	double maxValue = 0;
	Point maxLoc;
	minMaxLoc(darkChannelImg, NULL, &maxValue, NULL, &maxLoc);
	double A = min(minAtomsLight,maxValue);
	return A;
}

//---------------【Function3】getTransimissionImg()函数--------------------------//
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

//-------------------------【Function4】 Guidefilter函数-------------------------//
Mat GuidedFilter(Mat I_org, Mat p_org, int r, double eps, int s)
{
	/*
	主要参数说明:
	参数1: I_org 表示引导图,作为scrMat输入
	参数2: p_org 表示待滤波图像
	参数3: 滤波半径
	说明:本函数实现的是对单一通道进行的导向滤波
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
 
	r = (2 * r + 1) / s + 1;                                    //opencv中boxFilter()中的Size,比如9x9,我们就说半径为4   
 
	Mat mean_I;
	boxFilter(I, mean_I, CV_64FC1, Size(r, r));                 //这里计算 I 图的均值,也就是μ_k
 
	Mat mean_p;
	boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));             //这里计算 P 图的均值，也就是 p_k
  
	Mat mean_Ip;
	boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));     //这里计算 I乘P 的均值,a_k分母的第一项
   
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);                  //这个是 ak 的分子
	 
	Mat mean_II;
	boxFilter(I.mul(I), mean_II, CV_64FC1, Size(r, r));          //这里计算 I乘I 的均值, 作为计算方差的 EX^2
   
	Mat var_I = mean_II - mean_I.mul(mean_I);                    //得到I的方差
       
	Mat a = cov_Ip / (var_I + eps);
	Mat b = mean_p - a.mul(mean_I);

	Mat mean_a;
	boxFilter(a, mean_a, CV_64FC1, Size(r, r));                  //计算 ak 的均值
	Mat rmean_a;
	resize(mean_a, rmean_a, Size(I_org.cols, I_org.rows), 1);
   
	Mat mean_b;
	boxFilter(b, mean_b, CV_64FC1, Size(r, r));                   //计算 bk 的均值
	Mat rmean_b;
	resize(mean_b, rmean_b, Size(I_org.cols, I_org.rows), 1);
    
	Mat q = rmean_a.mul(_I) + rmean_b;
	Mat q1;
	q.convertTo(q1, CV_8UC1, 255, 0);
 
	return q1;
}

//-----------------------【funtion 6】getDehazedChanne-------------------------------//
Mat getDehazedChannel(Mat srcChannel, Mat transmissionChannel, double A)
{
	/*函数说明:
	函数功能: 本函数是用于计算单个通道的去雾图像
	参数1: 输入原图像的一个通道
	参数2: 传输图的单一通道(需要和原图像通道对应)
	参数3: 全球大气光值A
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

//-----------------【Function 7】getDehazedImg_guidedFilter-----------------------//
Mat getDehazedImg_guidedFilter(Mat src,Mat darkChannelImg)
{
	/*函数说明:
	本函数用于最终计算得到去雾效果图
	参数1: 输入的3通道图像
	参数2: 输入的暗通道图像
	*/

	Mat dehazedImg = Mat::zeros(src.rows, src.cols, CV_8UC3);             //构造一个3通道最终去雾图像
 
	Mat transmissionImg(src.rows, src.cols, CV_8UC3);
	Mat fineTransmissionImg(src.rows, src.cols, CV_8UC3);
	vector<Mat> srcChannel, dehazedChannel, transmissionChannel, fineTransmissionChannel;
 
	split(src, srcChannel);
	double A0 = getGlobelAtmosphericLight(darkChannelImg);
	double A1 = getGlobelAtmosphericLight(darkChannelImg);
	double A2 = getGlobelAtmosphericLight(darkChannelImg);
 
	split(transmissionImg, transmissionChannel);
	transmissionChannel.at(0) = getTransimissionImg(darkChannelImg, A0);                                    //得到每一个通道粗略的传输图
	transmissionChannel.at(1) = getTransimissionImg(darkChannelImg, A1);
	transmissionChannel.at(2) = getTransimissionImg(darkChannelImg, A2);
 
	split(fineTransmissionImg, fineTransmissionChannel);
	fineTransmissionChannel.at(0) = GuidedFilter(srcChannel.at(0), transmissionChannel.at(0), 64, 0.01, 8);  //经过导向滤波得到精细化的没一个通道的传输图
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
 
