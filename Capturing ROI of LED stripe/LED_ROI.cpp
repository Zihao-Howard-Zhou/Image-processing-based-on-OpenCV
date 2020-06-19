//--------------------------LED���Ƶ�ROI��ⷽ��-------------------------------//
/*����˵��:��demo�����˼���LED����ROI�Ļ�ȡ����:
1. ���ڻ���Բ�任��ʶ�𷽷�
2. ����findContours������������ҷ���
3. ��������ʽ˼·����(Ԥ������������ͬ)
*/

//-----------------------------ͷ�ļ���������---------------------------------//
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<string>


//----------------------------�����ռ�----------------------------------------//
using namespace cv;
using namespace std;

//-------------------------------------------������������----------------------------------------------------//
void method1(Mat& inputImage,int scale, int High_threshold = 100, int Low_threshold = 20, int kernel_size = 15);
void method2(Mat& inputImage, int kernel_size = 3);
void method3(Mat& input);

//----------------------------���������main----------------------------------//
int main()
{
	int method;
	Mat srcImage = imread("test2.jpg");  
	if(!srcImage.data )
	{
		cout << "��ȡʧ��!" << endl;
		return 0;
	}

	
	cout<<"-----------------------------------------"<<endl;
	cout<<"Please select the method you want to test"<<endl;
	cout<<"-----------------------------------------"<<endl;
	cout<<"------input: 1 ---> Enable method 1------"<<endl;
	cout<<"------input: 2 ---> Enable method 2------"<<endl;
	cout<<"------input: 3 ---> Enable method 3------"<<endl;
	cout<<"-----------------------------------------"<<endl;

	cin>>method;

	switch(method)
	{
	case 1: method1(srcImage, 30);
		break;
	case 2: method2(srcImage);
		break;
	case 3: method3(srcImage);
		break;
	default: cout<<"This function is temporarily unavailable!"<<endl;
		break;
	}	

	waitKey(0);
	return 0;
}



void method1(Mat& inputImage,int scale, int High_threshold, int Low_threshold, int kernel_size)
{
	/*method 1: ���ڻ���Բ�任��LED-ROI������ȡ
	����˵��:
	����1: �����Mat����ͼ��(������RGB��ͨ��ԭʼͼ)
	����2: scale,Ϊ����任��⵽��Բ��Բ��֮�����С����ĳ߶�,�������ǵ��㷨���������ֵ�������ͬԲ֮�����С����
	����3: canny�㷨���õ��ĸ���ֵ, Ĭ��ֵΪ100
	����4: canny�㷨���õ��ĵ���ֵ��Ĭ��ֵΪ20(ע��:����������ߴ��LEDʱ����ֵ������Ҫ��һ������)
	����5: ����˵ĳߴ�,�ڹ���������Զ����ں��н���ʹ��,Ĭ��ֵ15
	*/
	
	double t = (double)getTickCount();
	Mat closeImage, midImage;
	Mat element = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));    //�Զ����ں�
	morphologyEx(inputImage, closeImage, MORPH_CLOSE, element);                         //��������������LED����֮��ļ�϶
	
	cvtColor(closeImage, midImage, CV_BGR2GRAY);                             
	GaussianBlur(midImage, midImage, Size(9, 9), 2, 2);                                 //��˹�˲�������������

	vector<Vec3f> circles;
	HoughCircles(midImage, circles, CV_HOUGH_GRADIENT, 1, midImage.rows/30, 100, 20, 0, 0);  

	/*
	//------------------------------����ͨ��Բ��Ȧ��ROI����---------------------------------------//
	//������ͼ�л��Ƴ�Բ
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));                   //�����center����Բ�ĵ�
		//����, circles[i]��ʾ��i��Բ, circles[i][0]��ʾ��i��ԲԲ�ĵ�x����

		int radius = cvRound(circles[i][2]);                                            //�뾶r
		
		circle(srcImage, center, 3, Scalar(0, 0, 255), -1, 8, 0);                       //����Բ��
		
		circle(srcImage, center, radius, Scalar(0, 50, 255), 2, 8, 0);                  //����Բ����
	}
	*/

	vector<Mat> ROIs;
	Mat img_constructe(inputImage);

	//-----------------------------����ͨ�����ο�Ȧ��ROI����--------------------------------------//
	for(size_t j = 0; j < circles.size(); j++)
	{
		Rect rect(circles[j][0]-circles[j][2],circles[j][1]-circles[j][2],2*circles[j][2]+6,2*circles[j][2]+6);
	    //rectangle(srcImage,rect,Scalar(0,0,0),2,8);
		ROIs.push_back(inputImage(rect));
		rectangle(img_constructe, rect, Scalar(0, 255, 0), 2, 8);                      //�����Ҫ�ú���������ѡ�ô˾�
	}

	t = (double)getTickCount() - t;
	cout <<"mothod 1����ʱ��Ϊ:"<<1000 * t / (getTickFrequency()) << "ms" << endl;

	imshow("����ʶ�����", img_constructe);

	for(int i = 0; i < ROIs.size(); i++)
	{
		stringstream temp;
		temp<<i;
		string s1 = temp.str();
		imshow(s1, ROIs.at(i));
		//imwrite(s1+".jpg", ROIs.at(i));                                             //����ͼ��(optional)
	}
}


void method2(Mat& inputImage, int kernel_size)
{
	/*method2: ����findContours�������������
	����1: Mat���͵�����ͼ��(����ΪRGB3ͨ��ͼ)
	����2: int ���͵�kernel_size,��Ĭ��ֵΪ3
	*/

	double t = (double)getTickCount();
	Mat temp;
	inputImage.copyTo(temp);

	Mat element = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));             //�Զ����ں�
	morphologyEx(inputImage, inputImage, MORPH_CLOSE, element);                //��������������LED����֮��ļ�϶
	
	cvtColor(inputImage, inputImage, CV_BGR2GRAY);                             
	GaussianBlur(inputImage, inputImage, Size(3, 3), 2, 2);                    //��˹�˲�������������
 
	threshold(inputImage, inputImage, 10, 255, THRESH_BINARY);
	
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
 
	findContours(inputImage, contours , hierarchy , CV_RETR_CCOMP , CV_CHAIN_APPROX_SIMPLE );
	/*
	����4��������е�����,����������ֻ���������ȼ���ϵ,��ΧΪ����,����Χ�ڵ���Χ������������������������Ϣ,����Χ�ڵ����������������ڶ���
	����5�������������Ĺյ���Ϣ,�����������յ㴦�ĵ㱣����contours������,�յ���յ�֮��ֱ�߶��ϵ���Ϣ�㲻�豣��
    */

	vector<Point> maxcontours ;                                               //�������(�յ���Ϣ)
	double maxArea = 0;
 
	for( size_t i = 0; i < contours.size();i++ )                              //����ÿһ������
	{
		double area = contourArea( contours[i] );                             //contourArea��Ҫ���ڼ���ͼ�����������
		if( area > maxArea )
		{
			maxArea = area;
			maxcontours = contours[i];
		}
	}
 
	Rect maxRect = boundingRect( maxcontours );                               //���Ҿ��ο�, ����maxcounters��һ���㼯
	Mat result1;
 
	inputImage.copyTo(result1);
 
	for( size_t i = 0; i < contours.size(); i++ )
	{
		Rect r = boundingRect(contours[i]);
		r = r + Point(-2, -2);                                                //�����Ͻǵ���һ������(�����㹻��ԣ�ȱ���ͼ���в��ֱ��ض�)
		r = r + Size(3, 3);                                                   //���ţ����϶��㲻�䣬���+3���߶�+3
		//cout<<"���Ͻ�����:"<<r.tl();
		//cout<<"���εĿ��:"<<r.width;
		rectangle(result1 , r , Scalar(255) );
		rectangle(temp, r, Scalar(255), 1, 8, 0);                             

		//imshow( "all regions" , result1);
		//waitKey();
	}

	t = (double)getTickCount() - t;
	cout <<"mothod 2����ʱ��Ϊ:"<<1000 * t / (getTickFrequency()) << "ms" << endl;

	imshow( "all regions" , result1);
	imshow("����ʶ�����:", temp);  
	 
    //ֵ��ע�����:�˴������Ѿ����Ի������ROI���ο����Ϣ,��������ROI�Ͳ���������
}

void method3(Mat& input)
{
	double t = (double)getTickCount();
	Mat midimage1;
	Mat dstimage = input.clone();
	cvtColor(input, midimage1, COLOR_BGR2GRAY);
	GaussianBlur(midimage1, midimage1, Size(3, 3), 0);

	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));                 //��ýṹԪ��
	dilate(midimage1, midimage1, element);                                       //���Ͳ���,ȥ���ڲ�����
	
	threshold(midimage1, midimage1, 10, 255, THRESH_BINARY);                     //THRESH_BINARY_INV����ȡ����

	Canny(midimage1, midimage1, 50, 200, 5);

	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;

	//ֻ����߿򣬱���ȫ��ϸ��
	findContours(midimage1, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

	vector<Rect>boundRect(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(Mat(contours[i]));
	}

	for (unsigned int i = 0; i < boundRect.size(); i++)
	{
		rectangle(dstimage, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2, 8, 0);
	}
	t = (double)getTickCount() - t;
	cout <<"mothod 3����ʱ��Ϊ:"<<1000 * t / (getTickFrequency()) << "ms" << endl;


	imshow("Ч��ͼ", dstimage);
}


