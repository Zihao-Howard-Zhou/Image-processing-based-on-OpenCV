#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<string>


//----------------------------命名空间----------------------------------------//
using namespace cv;
using namespace std;

//-------------------------------------------函数声明部分----------------------------------------------------//
vector<Mat> ROIs;
void roi(Mat& input);
void  iterative_thresholding(Mat& rois);
vector<int> BitVector;


int main()
{

	Mat srcImage = imread("4.jpg");	
	roi(srcImage);
	cout << ROIs.at(1).rows << endl;
	iterative_thresholding(ROIs.at(0));
	
	waitKey(0);
	return 0;
}


void roi(Mat& input)
{
	double t = (double)getTickCount();
	Mat midimage1,midImage;
	Mat dstimage = input.clone();
	cvtColor(dstimage, midImage, COLOR_BGR2GRAY);
	cvtColor(input, midimage1, COLOR_BGR2GRAY);
	GaussianBlur(midimage1, midimage1, Size(3, 3), 0);

	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));                 //获得结构元素
	dilate(midimage1, midimage1, element);                                       //膨胀操作,去掉内部轮廓

	threshold(midimage1, midimage1, 20, 255, THRESH_BINARY);                     //THRESH_BINARY_INV可提取黑线

	Canny(midimage1, midimage1, 5, 200, 5);
	imshow("轮廓图", midimage1);
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;

	//只找外边框，保留全部细节
	findContours(midimage1, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

	vector<Rect>boundRect(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(Mat(contours[i]));
	}

	for (unsigned int i = 0; i < boundRect.size(); i++)
	{
		Rect rect(boundRect[i].tl().x, boundRect[i].tl().y, boundRect[i].width, boundRect[i].height);
		rectangle(dstimage, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2, 8, 0);
		ROIs.push_back(midImage(rect));
	}

	for (int i = 0; i < ROIs.size(); i++)
	{
		stringstream temp;
		temp << i;
		string s1 = temp.str();
		imshow(s1, ROIs.at(i));

	}

	imshow("效果图", dstimage);
}

void  iterative_thresholding(Mat& rois) {
	vector<float> grayworth;   //用向量存放T
	vector<Point> local_threshold;   //x代表阈值  ， 也代表第18列像素所在位置
	int count = 0;
	float worth = 0;

	for (int i = 0; i < rois.rows; i++) { //从开始寻找

		worth += rois.ptr<uchar>(18)[i];

		if ((i + 1) % 9 == 0) {                 //9个一组 算法开始
			count++;
			vector<float> R1;
			vector<float> R2;
			float avrag1 = 0;
			float avrag2 = 0;
			float Tk = 0;

			grayworth.push_back(worth / 9);
			do {     //当Tk不等于T时循环
				if (Tk != 0)
					grayworth[(i + 1) / 9 - 1] = Tk;   //将Tk赋值给T
				int s = i;
				for (s; s >= i - 8; s--) {
					if (rois.ptr<uchar>(18)[s] > grayworth[(i + 1) / 9 - 1])
						R1.push_back(rois.ptr<uchar>(18)[s]);  //R1
					else
						R2.push_back(rois.ptr<uchar>(18)[s]);   //R2
				}

				double sum = 0;
				for (size_t j = 0; j < R1.size(); j++) {    //计算U1
					sum += R1[j];
				}
				avrag1 = sum / R1.size();
				sum = 0;
				for (size_t j = 0; j < R2.size(); j++) {   //计算U2
					sum += R2[j];
				}
				avrag2 = sum / R2.size();
				Tk = (avrag1 + avrag2) / 2;    //Tk
			} while (Tk - grayworth[(i + 1) / 9 - 1] > -0.000001 &&Tk - grayworth[(i + 1) / 9 - 1]< 0.000001);

			//设置第n段阈值
			int n = i;
			for (n; n >= i - 8; n--) {
				local_threshold.push_back(Point(Tk, n));
			}
			worth = 0;
		}

		auto a = rois.rows % 9;
		if (i == rois.rows - 1 && a != 0) {     //对于最后不满9个一组的
			vector<float> R1;
			vector<float> R2;
			float avrag1 = 0;
			float avrag2 = 0;
			float Tk = 0;

			grayworth.push_back(worth / (rois.rows - count*9));
			do {     //当Tk不等于T时循环
				if (Tk != 0)
					grayworth[grayworth.size() - 1] = Tk;   //将Tk赋值给T
				int s = i;
				for (s; s >= count * 9; s--) {
					if (rois.ptr<uchar>(18)[s] > grayworth[grayworth.size() - 1])
						R1.push_back(rois.ptr<uchar>(18)[s]);  //R1
					else
						R2.push_back(rois.ptr<uchar>(18)[s]);   //R2
				}

				double sum = 0;
				for (size_t j = 0; j < R1.size(); j++) {    //计算U1
					sum += R1[j];
				}
				avrag1 = sum / R1.size();
				sum = 0;
				for (size_t j = 0; j < R2.size(); j++) {   //计算U2
					sum += R2[j];
				}
				avrag2 = sum / R2.size();
				Tk = (avrag1 + avrag2) / 2;    //Tk
			} while (Tk - grayworth[grayworth.size() - 1] > -0.000001 &&Tk - grayworth[grayworth.size() - 1]< 0.000001);

			//设置第n段阈值
			int n = i;
			for (n; n >= count * 9; n--) {
				local_threshold.push_back(Point(Tk, n));
			}
			worth = 0;
		}


	}
	cout << local_threshold << endl;
	
	for (int i = 0; i < rois.rows; i++) {
		if (rois.ptr<uchar>(18)[i] < local_threshold[i].x)
			BitVector.push_back(0);
		if (rois.ptr<uchar>(18)[i] >= local_threshold[i].x)
			BitVector.push_back(1);
	}
	for (int i = 0; i < rois.rows; i++) {
		cout << BitVector[i] << " ";
	}
}