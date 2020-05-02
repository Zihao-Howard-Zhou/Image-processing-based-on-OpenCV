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
	imshow("��˹��������0��", Gau_Pyr.at(0));
    imshow("��˹��������1��", Gau_Pyr.at(1));
	imshow("��˹��������2��", Gau_Pyr.at(2));
	imshow("��˹��������3��", Gau_Pyr.at(3));
	*/
	lp_Pyr = bulid_Laplacian_Pyr(Gau_Pyr, lp_Pyr, 3);
	lp_Pyr2 = bulid_Laplacian_Pyr(Gau_Pyr2, lp_Pyr2, 3);
	
	//imshow("������˹��������0��", lp_Pyr[0]);
	
	/*
	imshow("������˹��������1��", lp_Pyr.at(1));
	imshow("������˹��������2��", lp_Pyr.at(2));

*/


	
	Mat mask = Mat::zeros(height, width, CV_8UC3);     //������Ĥmask
	mask(Range::all(), Range(0, mask.cols * 0.5)) = 1.0;   //mask��������,Ȼ����벿����1,�Ұ벿����0

	
	vector<Mat> mask_Pyr, blend_lp;

	Mat result_higest_level;
	

	mask_Pyr = bulid_Gaussian_Pyr(mask, mask_Pyr, 3);
	/*
	cout<<"lp1������"<<lp_Pyr.back().type()<<endl;
	cout<<"mask������"<<(Scalar(1.0, 1.0, 1.0) - mask_Pyr.back()).type()<<endl;
	*/
	
	cout<<"1��С"<<Gau_Pyr.back().size()<<endl;
	cout<<"2��С"<<Gau_Pyr2.back().size()<<endl;
	waitKey();
	result_higest_level = Gau_Pyr.back().mul(mask_Pyr.back()) + ((Gau_Pyr2.back()).mul(Scalar(1.0, 1.0, 1.0) - mask_Pyr.back()));
	
	/*
	imshow("mask��˹�������ĵ�0��", mask_Pyr.at(0));
	imshow("mask��˹�������ĵ�1��", mask_Pyr.at(1));
	imshow("mask��˹�������ĵ�2��", mask_Pyr.at(2));
	imshow("mask��˹�������ĵ�3��", mask_Pyr.at(3));
	*/
	waitKey();

	blend_lp = blend_Laplacian_Pyr(lp_Pyr, lp_Pyr2, mask_Pyr, blend_lp);

	imshow("�ں�������˹�������ĵ�0��", blend_lp.at(0));
	imshow("�ں�������˹�������ĵ�1��", blend_lp.at(1));
	imshow("�ں�������˹�������ĵ�2��", blend_lp.at(2));
	imshow("ͼ���ںϵ����", result_higest_level);

	Mat output;
	output = blend(result_higest_level, blend_lp);
	imshow("�ں�Ч��ͼ", output);

	waitKey();
	return 0;


}

//-------------------Function 1 ��˹�������Ĺ���------------------------//
vector<Mat> bulid_Gaussian_Pyr(Mat& input, vector<Mat> Img_pyr, int level)
{
	/*
	����˵��:
	����1: ����� Mat ���ʹ����˹������ͼ��
	����2: ����ĸ�˹������(�� vector<Mat> ���ͱ���, ��ʹ��.at()��ȡĳһ�������)
	����3: ��˹�������ļ���( �˴�Ӧ�ر�ע��:��ʵ�Ĳ����� level + 1 !)
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

//---------------------------------Function 2 ������˹�������Ĺ���---------------------------------------------------------//
vector<Mat> bulid_Laplacian_Pyr(vector<Mat> Img_Gaussian_pyr, vector<Mat> Img_Laplacian_pyr, int level)
{
	/*
	����˵��:
	����1: ����ĸ�˹������ vector<Mat> ���ͣ�ÿһ��Ԫ�ش���ÿһ��
	����2: ������������˹������
	����3: ������˹�������Ĳ��� level
	*/
	vector<Mat> Img_Gaussian_pyr_temp;
	Img_Gaussian_pyr_temp.assign(Img_Gaussian_pyr.begin(), Img_Gaussian_pyr.end());   //����vector������ʹ��=�������˴�ʹ��assign���и���
	
	Mat for_sub, for_up;                  
	for(int i = 0; i < level; i++)
	{
		Mat for_up = Img_Gaussian_pyr_temp.back();       //��ȡ��˹��������ǰ��߲��ͼ�������
		Img_Gaussian_pyr_temp.pop_back();                //ɾ�����һ��Ԫ��
		
		for_sub = Img_Gaussian_pyr_temp.back();          //��ȡ������ͼ��

		Mat up2;
		pyrUp(for_up, up2, Size(for_up.cols * 2, for_up.rows * 2));    //�ϲ���

		Mat lp;
		/*
		cout<<"�ߴ�1"<<for_sub.size();
		cout<<"c�ߴ�2"<<up2.size();
		*/
		lp = for_sub - up2;
		Img_Laplacian_pyr.push_back(lp);
	}
	reverse(Img_Laplacian_pyr.begin(),Img_Laplacian_pyr.end());       //��һ�·�ת
	return Img_Laplacian_pyr;
}

vector<Mat> blend_Laplacian_Pyr(vector<Mat> Img1_lp, vector<Mat> Img2_lp,vector<Mat> mask_gau, vector<Mat> blend_lp)
{
	int level = Img1_lp.size();
	cout<<"level"<<level;   //3;
	//blend_lp.push_back((Img1_lp.back()).mul(mask_gau.back()) +
    //       (Img2_lp.back()).mul(Scalar(1.0, 1.0, 1.0) - mask_gau.back()));   //һ�ỹ��Ҫ��Img1, 2��˹�������Ķ�����а���mask��ӵĲ���
	for(int i = 0; i < level; i++)   //ע�� 0 ��ʾ����ͼ��˵���Ǵӵ׿�ʼ�ں�lp
	{  
		//(Img1_lp.at(i)).convertTo(Img1_lp.at(i), CV_32FC1); 
		//Mat Img1_lp_gray_temp;
		//cvtColor(Img1_lp.at(i), Img1_lp_gray_temp, CV_BGR2GRAY);
		//Mat A = Img1_lp_gray_temp.mul(mask_gau.at(i));
		Mat A = (Img1_lp.at(i)).mul(mask_gau.at(i));
		//imshow("ԭ����������˹������i��ͼ", Img1_lp.at(i));
		//imshow("A", A);
		//waitKey();
		/*
		cout<<"1����"<<Img1_lp.at(i).type()<<endl;
		cout<<"2����"<<mask_gau.at(i).type()<<endl;
		waitKey();
		*/
		//cout<<"mask�Ĵ�С"<<mask_gau[i].size();
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
