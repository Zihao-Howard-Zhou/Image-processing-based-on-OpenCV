#include <iostream>    
#include <opencv2\opencv.hpp>    
#include <opencv2\highgui\highgui.hpp>    
using namespace std;
using namespace cv;
 
Mat MultiScaleDetailBoosting(Mat src, int Radius);

int main(int argc)
{
	Mat src = imread("test11.jpg",1);
	imshow("src", src);
 
	Mat dest=MultiScaleDetailBoosting(src,9);
	imshow("dest", dest);
 
	cvWaitKey();
	return 0;
}

Mat MultiScaleDetailBoosting(Mat src, int Radius)
{
    int rows = src.rows;
    int cols = src.cols;
    Mat B1, B2, B3;

    GaussianBlur(src, B1, Size(Radius, Radius), 1.0, 1.0);           //高斯模糊, 高斯内核在X方向的标准偏差为1.0，高斯内核在Y方向的标准偏差也是1.0
    GaussianBlur(src, B2, Size(Radius*2-1, Radius*2-1), 2.0, 2.0);   //高斯内核在X方向的标准偏差为2.0，高斯内核在Y方向的标准偏差为2.0
    GaussianBlur(src, B3, Size(Radius*4-1, Radius*4-1), 4.0, 4.0);
	

    float w1 = 0.5, w2 = 0.5, w3 = 0.25;                             //论文给出

    Mat dst(rows, cols, CV_8UC3);

    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
	{
            for(int k = 0; k < 3; k++)
	    {
                int D1 = src.at<Vec3b>(i, j)[k] - B1.at<Vec3b>(i, j)[k];  //D1 = I* - B1
                int D2 = B1.at<Vec3b>(i, j)[k] - B2.at<Vec3b>(i, j)[k];   //D2 = B1 - B2
                int D3 = B2.at<Vec3b>(i, j)[k] - B3.at<Vec3b>(i, j)[k];   //D3 = B2 - B3

                int sign = D1 > 0 ? 1 : -1;

                dst.at<Vec3b>(i, j)[k] = saturate_cast<uchar>((1 - w1*sign) * D1 + w2 * D2 + w3 * D3 + src.at<Vec3b>(i, j)[k]);
            }
        }
    }
    return dst;
}

