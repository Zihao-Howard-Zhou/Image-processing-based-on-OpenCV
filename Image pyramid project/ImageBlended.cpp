//----------------------------基于拉普拉斯金字塔的图像融合算法---------------------------//
//Configuration: Visual Studio 2010 + Opencv2.4.10
//Author: ZiHao Zhou / South China University of Technology 
//Date: 2020.5.2

#include <opencv2/opencv.hpp>
#include<vector>
#include<iostream>


using namespace std;
using namespace cv;

vector<Mat> bulid_Gaussian_Pyr(Mat& input, vector<Mat> Img_pyr, int level);
vector<Mat> bulid_Laplacian_Pyr(vector<Mat> Img_Gaussian_pyr, vector<Mat> Img_Laplacian_pyr, int level);
vector<Mat> blend_Laplacian_Pyr(vector<Mat> Img1_lp, vector<Mat> Img2_lp,vector<Mat> mask_gau, vector<Mat> blend_lp);
Mat blend(Mat& result_higest_level, vector<Mat> blend_lp);

int main()
{
	vector<Mat> Gau_Pyr, lp_Pyr, Gau_Pyr2, lp_Pyr2;
	vector<Mat> maskGaussianPyramid; 

	Mat input = imread("4.jpg", 1);
	Mat input2 = imread("test11.jpg", 1);

	imshow("待融合图像1", input);
	imshow("待融合图像2", input2);

	resize(input,input,Size(600, 600));
	
	int height = input.rows;
	int width = input.cols;
	//cout<<"width"<<width<<endl;
	//cout<<"height"<<height<<endl;

	resize(input2, input2, Size(600,600));

	input.convertTo(input, CV_32F);                  //转换成CV_32F, 用于和mask的类型匹配( 另外 CV_32F 类型精度高, 有利于计算)
	input2.convertTo(input2, CV_32F);

	Gau_Pyr = bulid_Gaussian_Pyr(input, Gau_Pyr,3);       //计算两张图片的高斯金字塔
	Gau_Pyr2 = bulid_Gaussian_Pyr(input2, Gau_Pyr2,3);
	
	/*
	imshow("高斯金字塔第0层", Gau_Pyr.at(0));
        imshow("高斯金字塔第1层", Gau_Pyr.at(1));
	imshow("高斯金字塔第2层", Gau_Pyr.at(2));
	imshow("高斯金字塔第3层", Gau_Pyr.at(3));
	*/
	lp_Pyr = bulid_Laplacian_Pyr(Gau_Pyr, lp_Pyr, 3);     //计算两者图像的拉普拉斯金字塔
	lp_Pyr2 = bulid_Laplacian_Pyr(Gau_Pyr2, lp_Pyr2, 3);
	
	/*
	imshow("拉普拉斯金字塔第0层", lp_Pyr.at(0));    //当然,使用blend_lp[0]也是可以的
	imshow("拉普拉斯金字塔第1层", lp_Pyr.at(1));
	imshow("拉普拉斯金字塔第2层", lp_Pyr.at(2));
        */

	
	Mat mask = Mat::zeros(height, width, CV_32FC1);           //构造掩膜mask, CV_32FC1类型, 大小和 Img1 一样
	mask(Range::all(), Range(0, mask.cols * 0.5)) = 1.0;      //mask的所有行,然后左半部分是1,右半部分是0 (意思是对半融合)

	cvtColor(mask, mask, CV_GRAY2BGR);                        //因为此时mask是单channel的,Img是3channel的,所以还需要cvtColor

	//cout<<"现在mask的type"<<mask.type()<<endl;
	//cout<<"现在的lp的type"<<lp_Pyr.at(0).type()<<endl;

	vector<Mat> mask_Pyr, blend_lp;

	Mat result_higest_level;                                  //图像融合的起点
	
	mask_Pyr = bulid_Gaussian_Pyr(mask, mask_Pyr, 3);         //mask的高斯金字塔也是 level+1 层的
	
	//下面将 Img1, Img2的高斯金字塔的顶层按照mask融合
	result_higest_level = Gau_Pyr.back().mul(mask_Pyr.back()) + ((Gau_Pyr2.back()).mul(Scalar(1.0, 1.0, 1.0) - mask_Pyr.back()));
	
	/*
	imshow("mask高斯金字塔的第0层", mask_Pyr.at(0));
	imshow("mask高斯金字塔的第1层", mask_Pyr.at(1));
	imshow("mask高斯金字塔的第2层", mask_Pyr.at(2));
	imshow("mask高斯金字塔的第3层", mask_Pyr.at(3));
	*/

	blend_lp = blend_Laplacian_Pyr(lp_Pyr, lp_Pyr2, mask_Pyr, blend_lp);

	/*
	imshow("融合拉普拉斯金字塔的第0层", blend_lp.at(0));
	imshow("融合拉普拉斯金字塔的第1层", blend_lp.at(1));
	imshow("融合拉普拉斯金字塔的第2层", blend_lp.at(2));
	*/
	imshow("图像融合的起点", result_higest_level);

	Mat output;
	output = blend(result_higest_level, blend_lp);
	output.convertTo(output, CV_8UC3);
	imshow("融合效果图", output);

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
	reverse(Img_Laplacian_pyr.begin(),Img_Laplacian_pyr.end());       //做一下反转(0->最大尺寸的金字塔层)
	return Img_Laplacian_pyr;
}

//------------------------------------Function 3 混合拉普拉斯金字塔的构建-----------------------------------------//
vector<Mat> blend_Laplacian_Pyr(vector<Mat> Img1_lp, vector<Mat> Img2_lp,vector<Mat> mask_gau, vector<Mat> blend_lp)
{
	/*参数说明:
	参数1: 待融合图像1的拉普拉斯金字塔 vector<Mat> 类型(level 层)
	参数2: 待融合图像2的拉普拉斯金字塔 vector<Mat> 类型
	参数3: mask掩膜的高斯金字塔 (level+1层)
	参数4: 待返回的混合拉普拉斯金字塔 vector<Mat> 类型
	*/

	int level = Img1_lp.size();

	//cout<<"level"<<level;  确认level级数 

	for(int i = 0; i < level; i++)                                        //注意 0 表示最大的图，说明是从底开始融合lp
	{  
		Mat A = (Img1_lp.at(i)).mul(mask_gau.at(i));                      //根据mask(作为权重) 
		
		Mat antiMask = Scalar(1.0, 1.0, 1.0) - mask_gau[i];
		Mat B = Img2_lp[i].mul(antiMask);
		Mat blendedLevel = A + B;                                         //待融合图像的拉普拉斯金字塔对应层按照mask融合
		blend_lp.push_back(blendedLevel);                                 //存入blend_lp, 作为第 i 层
	}
	return blend_lp;
}

//--------------Function 4 图像融合---------------------//
Mat blend(Mat& result_higest_level, vector<Mat> blend_lp)
{
	/*参数说明:
	参数1: 图像混合的起点Mat (也就是两个带融合图像高斯金字塔最高层按mask加权求和的结果
	参数2: Function 3所求得的混合拉普拉斯金字塔 vector<Mat> 类型
	*/

	int level = blend_lp.size();      
	Mat for_up, temp_add;
	for(int i = 0; i < level; i++)
	{	
		pyrUp(result_higest_level, for_up, Size(result_higest_level.cols * 2, result_higest_level.rows * 2));
		temp_add = blend_lp.back() + for_up;
		blend_lp.pop_back();              //因为此处是直接删除最后一个元素,所以在调用本函数之前如后续的代码还需要blend_lp, 需要先行保存拷贝
		result_higest_level = temp_add;
	}
	return temp_add;
}
