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
	
	imshow("��˹��������0��", left_Pyr.at(0));
    imshow("��˹��������1��", left_Pyr.at(1));
	imshow("��˹��������2��", left_Pyr.at(2));
	imshow("��˹��������3��", left_Pyr.at(3));
	
	lp_Pyr = bulid_Laplacian_Pyr(left_Pyr, lp_Pyr, 3);
	imshow("������˹��������0��", lp_Pyr.at(0));
	imshow("������˹��������1��", lp_Pyr.at(1));
	imshow("������˹��������2��", lp_Pyr.at(2));


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
