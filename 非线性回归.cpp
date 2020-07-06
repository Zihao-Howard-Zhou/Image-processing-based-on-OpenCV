#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/core.hpp>

using namespace std;
using namespace cv;
//�����Իع��㷨
//���ʵ��ֵ-Ԥ��ֵ ��ƽ��������ݶ���Сֵ�õ�����
//ͨ������ʵ�����Ч���Ľ�
//�ʼ���Ǹ������Լ�Ϲ����
//theta = theta - alpha * gradient  #�����·���Ĺ�ʽ���� = �� - ����(h(x) - y)x
//��С�ݶȷ�ֻ��ȷ�������߻���������ȥ�ƽ�
//�������ʼ���������ģ�����ȷ����
//��ƫ������Ҫ���Լ���������ʽ

double sum1, sum2, sum3, sum4;
double sumy;
double a0 , a1 , a2,a3,a4 ;
double alpha= 0.0004;
Mat polyfit(vector<Point>& , int );
//double gradienta0(int );
//double gradienta1(int );
//double gradienta2(int );
double result(int );
int main()
{
	Mat src = imread("coding test.jpg");
	Mat mid;
	cvtColor(src, mid, COLOR_BGR2GRAY);
	Mat src2;
	Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));
	dilate(src, src2, element);
	Mat src3;
	cvtColor(src2, src3, COLOR_BGR2GRAY);
	Mat edges;
	Canny(src3, edges, 100, 100);
	vector<vector<Point>>contours;
	vector<Vec4i> hierarchy;
	findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	
	vector<Rect>rect;
	for (int i = 0; i < contours.size(); i++)
	{
		rect.push_back(boundingRect(contours[i]));
	}	
	vector<Mat>imageroi;
	for (int i = 0; i < rect.size(); i++)
	{
		imageroi.push_back(mid(rect[i]));
	}
	imshow("��Ҫ������ͼ", imageroi[3]);
	Mat src1 = imageroi[3].col(imageroi[3].size().height / 2);
	a0 = 1, a2 = 1, a1 = 1;
	sum1 = 0; sum2 = 0; sum3 = 0; sum4 = 0; sumy = 0;
	vector<Point>in_point;
	for (int i = 0; i <= src1.rows - 1; i++)
	{
		int y = src1.at<uchar>(i, 0);
		in_point.push_back(Point(i, y));
	}
	/*	cout << y<<'\t';
		sum1 += i;
		sum2 += i * i;
		sum3 += i * i * i;
		sum4 += i * i * i * i;
		sumy += y;
	}
	cout << sum1 << '\t' << sum2 << '\t' << sum3 << '\t' << sum4 << endl;
		cout << sumy << endl;
	for (int i = 0; i < 100; i++)
	{
		a0 = a0 - alpha * gradienta0(src1.rows);
		a1 = a1 - alpha * gradienta1(src1.rows);
		a2 = a2 - alpha * gradienta2(src1.rows);
	}*/

	Mat resul = polyfit(in_point, 4);
	a0 = resul.at<double>(0, 0);
	a1 = resul.at<double>(1, 0);
	a2 = resul.at<double>(2, 0);
	a3 = resul.at<double>(3, 0);
	a4 = resul.at<double>(4, 0);

	for (int i = 0; i < in_point.size(); i++)
	{
		if (in_point[i].y > result(i))
		{
			cout << 1;
		}
		else
			cout << 0;
	}
	return 0;
}
//double gradienta0(int m)
//{
//	double gradient;
//	gradient =( a0 - sumy + (double)a1 * sum1 + (double)a2 * sum2)/m;
//	if (gradient > 0)
//		return gradient;
//	else
//		return -gradient;
//}
//double gradienta1(int m)
//{
//	double gradient;
//	gradient= (a1 * (double)sum2 -  sumy * sum1 + (double)sum3 * a2 + (double)sum1 * a0)/m;
//	
//	if (gradient > 0)
//		return gradient;
//	else
//		return -gradient; 
//}
//double gradienta2(int m)
//{
//	double gradient;
//	gradient =(a2*(double)sum4 - sumy*sum2 + a1 * (double)sum3+ a2 * (double)sum2+ a0*(double)sum2)/m;
//	if (gradient > 0)
//		return gradient; 
//	else
//		return -gradient;
//}
double result(int x)
{
	double result;
	result = a0 + a1 * x + a2 * x * x+a3*x*x*x+a4*x*x*x*x;
	return result;
}

Mat polyfit(vector<Point>& in_point, int n)
{
	int size = in_point.size();
	//����δ֪������
	int x_num = n + 1;
	//�������U��Y
	Mat mat_u(size, x_num, CV_64F);
	Mat mat_y(size, 1, CV_64F);

	for (int i = 0; i < mat_u.rows; ++i)
		for (int j = 0; j < mat_u.cols; ++j)
		{
			mat_u.at<double>(i, j) = pow(in_point[i].x, j);
		}

	for (int i = 0; i < mat_y.rows; ++i)
	{
		mat_y.at<double>(i, 0) = in_point[i].y;
	}

	//�������㣬���ϵ������K
	Mat mat_k(x_num, 1, CV_64FC1);
	mat_k = (mat_u.t() * mat_u).inv() * mat_u.t() * mat_y;//��c++������������
	cout << mat_k << endl;
	return mat_k;
}