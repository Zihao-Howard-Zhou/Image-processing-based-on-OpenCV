#include <opencv2/opencv.hpp>
#include<vector>
#include<iostream>


using namespace std;
using namespace cv;

vector<Mat> bulid_Gaussian_Pyr(Mat& input, vector<Mat> Img_pyr, int level);
vector<Mat> bulid_Laplacian_Pyr(vector<Mat> Img_Gaussian_pyr, vector<Mat> Img_Laplacian_pyr, int level);

int main()
{
	vector<Mat> left_Pyr, Right_Pyr, Rescult_Pyr, lp_Pyr;
	vector<Mat> maskGaussianPyramid; 

	Mat input = imread("4.jpg", 1);
	resize(input,input,Size(400, 400));

	left_Pyr = bulid_Gaussian_Pyr(input, left_Pyr,3);
	
	imshow("高斯金字塔第0层", left_Pyr.at(0));
    imshow("高斯金字塔第1层", left_Pyr.at(1));
	imshow("高斯金字塔第2层", left_Pyr.at(2));
	imshow("高斯金字塔第3层", left_Pyr.at(3));
	
	lp_Pyr = bulid_Laplacian_Pyr(left_Pyr, lp_Pyr, 3);
	imshow("拉普拉斯金字塔第0层", lp_Pyr.at(0));
	imshow("拉普拉斯金字塔第1层", lp_Pyr.at(1));
	imshow("拉普拉斯金字塔第2层", lp_Pyr.at(2));


	waitKey();
	return 0;


}

//-------------------Function 1 高斯金字塔的构建------------------------//
vector<Mat> bulid_Gaussian_Pyr(Mat& input, vector<Mat> Img_pyr, int level)
{
	/*
	参数说明:
	参数1: 输入的 Mat 类型待求高斯金字塔图像
	参数2: 输出的高斯金字塔(以 vector<Mat> 类型保存, 可使用.at()获取某一层的内容)
	参数3: 高斯金字塔的级数( 此处应特别注意:真实的层数是 level + 1 !)
	*/
	Img_pyr.push_back(input);
	Mat dst;
	for(int i = 0; i < level; i++)
	{
		pyrDown(input, dst, Size(input.cols/2, input.rows/2));
		Img_pyr.push_back(dst);
		input = dst;
	}
	return Img_pyr;
}

//---------------------------------Function 2 拉普拉斯金字塔的构建---------------------------------------------------------//
vector<Mat> bulid_Laplacian_Pyr(vector<Mat> Img_Gaussian_pyr, vector<Mat> Img_Laplacian_pyr, int level)
{
	/*
	参数说明:
	参数1: 输入的高斯金字塔 vector<Mat> 类型，每一个元素代表每一层
	参数2: 待求解的拉普拉斯金字塔
	参数3: 拉普拉斯金字塔的层数 level
	*/
	vector<Mat> Img_Gaussian_pyr_temp;
	Img_Gaussian_pyr_temp.assign(Img_Gaussian_pyr.begin(), Img_Gaussian_pyr.end());   //由于vector对象不能使用=拷贝，此处使用assign进行复制
	
	Mat for_sub, for_up;                  
	for(int i = 0; i < level; i++)
	{
		Mat for_up = Img_Gaussian_pyr_temp.back();       //获取高斯金字塔当前最高层的图像的引用
		Img_Gaussian_pyr_temp.pop_back();                //删除最后一个元素
		
		for_sub = Img_Gaussian_pyr_temp.back();          //获取被减数图像

		Mat up2;
		pyrUp(for_up, up2, Size(for_up.cols * 2, for_up.rows * 2));    //上采样

		Mat lp;
		/*
		cout<<"尺寸1"<<for_sub.size();
		cout<<"c尺寸2"<<up2.size();
		*/
		lp = for_sub - up2;
		Img_Laplacian_pyr.push_back(lp);
	}
	reverse(Img_Laplacian_pyr.begin(),Img_Laplacian_pyr.end());       //做一下反转
	return Img_Laplacian_pyr;
}
