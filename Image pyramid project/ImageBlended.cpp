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

	resize(input,input,Size(400, 400));
	
	int height = input.rows;
	int width = input.cols;
	cout<<"width"<<width<<endl;
	cout<<"height"<<height<<endl;

	resize(input2, input2, Size(400,400));

	Gau_Pyr = bulid_Gaussian_Pyr(input, Gau_Pyr,3);
	Gau_Pyr2 = bulid_Gaussian_Pyr(input2, Gau_Pyr2,3);
	
	/*
	imshow("高斯金字塔第0层", Gau_Pyr.at(0));
    imshow("高斯金字塔第1层", Gau_Pyr.at(1));
	imshow("高斯金字塔第2层", Gau_Pyr.at(2));
	imshow("高斯金字塔第3层", Gau_Pyr.at(3));
	*/
	lp_Pyr = bulid_Laplacian_Pyr(Gau_Pyr, lp_Pyr, 3);
	lp_Pyr2 = bulid_Laplacian_Pyr(Gau_Pyr2, lp_Pyr2, 3);
	
	//imshow("拉普拉斯金字塔第0层", lp_Pyr[0]);
	
	/*
	imshow("拉普拉斯金字塔第1层", lp_Pyr.at(1));
	imshow("拉普拉斯金字塔第2层", lp_Pyr.at(2));

*/


	
	Mat mask = Mat::zeros(height, width, CV_8UC3);     //构造掩膜mask
	mask(Range::all(), Range(0, mask.cols * 0.5)) = 1.0;   //mask的所有行,然后左半部分是1,右半部分是0

	
	vector<Mat> mask_Pyr, blend_lp;

	Mat result_higest_level;
	

	mask_Pyr = bulid_Gaussian_Pyr(mask, mask_Pyr, 3);
	/*
	cout<<"lp1的类型"<<lp_Pyr.back().type()<<endl;
	cout<<"mask的类型"<<(Scalar(1.0, 1.0, 1.0) - mask_Pyr.back()).type()<<endl;
	*/
	
	cout<<"1大小"<<Gau_Pyr.back().size()<<endl;
	cout<<"2大小"<<Gau_Pyr2.back().size()<<endl;
	waitKey();
	result_higest_level = Gau_Pyr.back().mul(mask_Pyr.back()) + ((Gau_Pyr2.back()).mul(Scalar(1.0, 1.0, 1.0) - mask_Pyr.back()));
	
	/*
	imshow("mask高斯金字塔的第0层", mask_Pyr.at(0));
	imshow("mask高斯金字塔的第1层", mask_Pyr.at(1));
	imshow("mask高斯金字塔的第2层", mask_Pyr.at(2));
	imshow("mask高斯金字塔的第3层", mask_Pyr.at(3));
	*/
	waitKey();

	blend_lp = blend_Laplacian_Pyr(lp_Pyr, lp_Pyr2, mask_Pyr, blend_lp);

	imshow("融合拉普拉斯金字塔的第0层", blend_lp.at(0));
	imshow("融合拉普拉斯金字塔的第1层", blend_lp.at(1));
	imshow("融合拉普拉斯金字塔的第2层", blend_lp.at(2));
	imshow("图像融合的起点", result_higest_level);

	Mat output;
	output = blend(result_higest_level, blend_lp);
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
	reverse(Img_Laplacian_pyr.begin(),Img_Laplacian_pyr.end());       //做一下反转
	return Img_Laplacian_pyr;
}

vector<Mat> blend_Laplacian_Pyr(vector<Mat> Img1_lp, vector<Mat> Img2_lp,vector<Mat> mask_gau, vector<Mat> blend_lp)
{
	int level = Img1_lp.size();
	cout<<"level"<<level;   //3;
	//blend_lp.push_back((Img1_lp.back()).mul(mask_gau.back()) +
    //       (Img2_lp.back()).mul(Scalar(1.0, 1.0, 1.0) - mask_gau.back()));   //一会还需要把Img1, 2高斯金字塔的顶层进行按照mask相加的操作
	for(int i = 0; i < level; i++)   //注意 0 表示最大的图，说明是从底开始融合lp
	{  
		//(Img1_lp.at(i)).convertTo(Img1_lp.at(i), CV_32FC1); 
		//Mat Img1_lp_gray_temp;
		//cvtColor(Img1_lp.at(i), Img1_lp_gray_temp, CV_BGR2GRAY);
		//Mat A = Img1_lp_gray_temp.mul(mask_gau.at(i));
		Mat A = (Img1_lp.at(i)).mul(mask_gau.at(i));
		//imshow("原来的拉普拉斯金字塔i层图", Img1_lp.at(i));
		//imshow("A", A);
		//waitKey();
		/*
		cout<<"1类型"<<Img1_lp.at(i).type()<<endl;
		cout<<"2类型"<<mask_gau.at(i).type()<<endl;
		waitKey();
		*/
		//cout<<"mask的大小"<<mask_gau[i].size();
		Mat antiMask = Scalar(1.0, 1.0, 1.0) - mask_gau[i];
		Mat B = Img2_lp[i].mul(antiMask);
		Mat blendedLevel = A + B;
		blend_lp.push_back(blendedLevel);
	}
	return blend_lp;
}

Mat blend(Mat& result_higest_level, vector<Mat> blend_lp)
{
	int level = blend_lp.size();
	Mat for_up, temp_add;
	for(int i = 0; i < level; i++)
	{	
		pyrUp(result_higest_level, for_up, Size(result_higest_level.cols * 2, result_higest_level.rows * 2));
		temp_add = blend_lp.back() + for_up;
		blend_lp.pop_back();
		result_higest_level = temp_add;
	}
	return temp_add;
}
