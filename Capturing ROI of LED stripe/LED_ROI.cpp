//--------------------------LED条纹的ROI检测方法-------------------------------//
/*程序说明:本demo给出了几种LED条纹ROI的获取方法:
1. 基于霍夫圆变换的识别方法
2. 基于findContours的最大轮廓查找方法
3. 与上述方式思路类似(预处理步骤有所不同)
*/

//-----------------------------头文件声明部分---------------------------------//
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<string>


//----------------------------命名空间----------------------------------------//
using namespace cv;
using namespace std;

//-------------------------------------------函数声明部分----------------------------------------------------//
void method1(Mat& inputImage,int scale, int High_threshold = 100, int Low_threshold = 20, int kernel_size = 15);
void method2(Mat& inputImage, int kernel_size = 3);
void method3(Mat& input);

//----------------------------主程序入口main----------------------------------//
int main()
{
	int method;
	Mat srcImage = imread("test2.jpg");  
	if(!srcImage.data )
	{
		cout << "读取失败!" << endl;
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
	/*method 1: 基于霍夫圆变换的LED-ROI区域提取
	参数说明:
	参数1: 输入的Mat类型图像(允许是RGB三通道原始图)
	参数2: scale,为霍夫变换检测到的圆的圆心之间的最小距离的尺度,即让我们的算法能明显区分的两个不同圆之间的最小距离
	参数3: canny算法所用到的高阈值, 默认值为100
	参数4: canny算法所用到的低阈值，默认值为20(注意:在面对其他尺寸的LED时，该值可能需要进一步调整)
	参数5: 卷积核的尺寸,在构造闭运算自定义内核中将会使用,默认值15
	*/
	
	double t = (double)getTickCount();
	Mat closeImage, midImage;
	Mat element = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));    //自定义内核
	morphologyEx(inputImage, closeImage, MORPH_CLOSE, element);                         //闭运算用于消除LED条纹之间的间隙
	
	cvtColor(closeImage, midImage, CV_BGR2GRAY);                             
	GaussianBlur(midImage, midImage, Size(9, 9), 2, 2);                                 //高斯滤波处理消除噪声

	vector<Vec3f> circles;
	HoughCircles(midImage, circles, CV_HOUGH_GRADIENT, 1, midImage.rows/30, 100, 20, 0, 0);  

	/*
	//------------------------------下面通过圆形圈出ROI区域---------------------------------------//
	//依次在图中绘制出圆
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));                   //这里的center就是圆心点
		//其中, circles[i]表示第i个圆, circles[i][0]表示第i个圆圆心的x坐标

		int radius = cvRound(circles[i][2]);                                            //半径r
		
		circle(srcImage, center, 3, Scalar(0, 0, 255), -1, 8, 0);                       //绘制圆心
		
		circle(srcImage, center, radius, Scalar(0, 50, 255), 2, 8, 0);                  //绘制圆轮廓
	}
	*/

	vector<Mat> ROIs;
	Mat img_constructe(inputImage);

	//-----------------------------下面通过矩形框圈出ROI区域--------------------------------------//
	for(size_t j = 0; j < circles.size(); j++)
	{
		Rect rect(circles[j][0]-circles[j][2],circles[j][1]-circles[j][2],2*circles[j][2]+6,2*circles[j][2]+6);
	    //rectangle(srcImage,rect,Scalar(0,0,0),2,8);
		ROIs.push_back(inputImage(rect));
		rectangle(img_constructe, rect, Scalar(0, 255, 0), 2, 8);                      //如果需要用红框框起来就选用此句
	}

	t = (double)getTickCount() - t;
	cout <<"mothod 1所用时间为:"<<1000 * t / (getTickFrequency()) << "ms" << endl;

	imshow("整体识别情况", img_constructe);

	for(int i = 0; i < ROIs.size(); i++)
	{
		stringstream temp;
		temp<<i;
		string s1 = temp.str();
		imshow(s1, ROIs.at(i));
		//imwrite(s1+".jpg", ROIs.at(i));                                             //保存图像(optional)
	}
}


void method2(Mat& inputImage, int kernel_size)
{
	/*method2: 基于findContours的最大轮廓查找
	参数1: Mat类型的输入图像(允许为RGB3通道图)
	参数2: int 类型的kernel_size,有默认值为3
	*/

	double t = (double)getTickCount();
	Mat temp;
	inputImage.copyTo(temp);

	Mat element = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));             //自定义内核
	morphologyEx(inputImage, inputImage, MORPH_CLOSE, element);                //闭运算用于消除LED条纹之间的间隙
	
	cvtColor(inputImage, inputImage, CV_BGR2GRAY);                             
	GaussianBlur(inputImage, inputImage, Size(3, 3), 2, 2);                    //高斯滤波处理消除噪声
 
	threshold(inputImage, inputImage, 10, 255, THRESH_BINARY);
	
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
 
	findContours(inputImage, contours , hierarchy , CV_RETR_CCOMP , CV_CHAIN_APPROX_SIMPLE );
	/*
	参数4：检测所有的轮廓,但所有轮廓只建立两个等级关系,外围为顶层,若外围内的内围轮廓还包含了其他的轮廓信息,则内围内的所有轮廓均归属于顶层
	参数5：仅保存轮廓的拐点信息,把所有轮廓拐点处的点保存入contours向量内,拐点与拐点之间直线段上的信息点不予保留
    */

	vector<Point> maxcontours ;                                               //最大轮廓(拐点信息)
	double maxArea = 0;
 
	for( size_t i = 0; i < contours.size();i++ )                              //遍历每一个轮廓
	{
		double area = contourArea( contours[i] );                             //contourArea主要用于计算图像轮廓的面积
		if( area > maxArea )
		{
			maxArea = area;
			maxcontours = contours[i];
		}
	}
 
	Rect maxRect = boundingRect( maxcontours );                               //查找矩形框, 参数maxcounters是一个点集
	Mat result1;
 
	inputImage.copyTo(result1);
 
	for( size_t i = 0; i < contours.size(); i++ )
	{
		Rect r = boundingRect(contours[i]);
		r = r + Point(-2, -2);                                                //对左上角点做一个缩进(留有足够的裕度避免图像有部分被截断)
		r = r + Size(3, 3);                                                   //缩放，左上顶点不变，宽度+3，高度+3
		//cout<<"左上角坐标:"<<r.tl();
		//cout<<"矩形的宽度:"<<r.width;
		rectangle(result1 , r , Scalar(255) );
		rectangle(temp, r, Scalar(255), 1, 8, 0);                             

		//imshow( "all regions" , result1);
		//waitKey();
	}

	t = (double)getTickCount() - t;
	cout <<"mothod 2所用时间为:"<<1000 * t / (getTickFrequency()) << "ms" << endl;

	imshow( "all regions" , result1);
	imshow("整体识别情况:", temp);  
	 
    //值得注意的是:此处我们已经可以获得所有ROI矩形框的信息,单独画出ROI就不是问题了
}

void method3(Mat& input)
{
	double t = (double)getTickCount();
	Mat midimage1;
	Mat dstimage = input.clone();
	cvtColor(input, midimage1, COLOR_BGR2GRAY);
	GaussianBlur(midimage1, midimage1, Size(3, 3), 0);

	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));                 //获得结构元素
	dilate(midimage1, midimage1, element);                                       //膨胀操作,去掉内部轮廓
	
	threshold(midimage1, midimage1, 10, 255, THRESH_BINARY);                     //THRESH_BINARY_INV可提取黑线

	Canny(midimage1, midimage1, 50, 200, 5);

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
		rectangle(dstimage, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2, 8, 0);
	}
	t = (double)getTickCount() - t;
	cout <<"mothod 3所用时间为:"<<1000 * t / (getTickFrequency()) << "ms" << endl;


	imshow("效果图", dstimage);
}


