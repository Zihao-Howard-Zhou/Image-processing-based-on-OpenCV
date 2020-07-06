/*--------------------------------------------------------------------------------------
		   本代码是使用EVA实现LED的数据逻辑检测
代码日志:
作者:ZZH South China University of Technology
时间: 2020年6月29日 17:34

版本: 1.0.0

目前代码可实现的功能:完成了EVA算法,可以得到采样之后的判决结果

暂时未实现的功能: 查找header,进一步提取ID
拟采用的方案: 由于header是010101这样的序列,故可以采样简单的模板匹配去找010101的部分

待解决的BUG: 对于第二个ROI的提取,左边总是多出一小块黑色部分,还需要思考怎么去除
拟采用方案: 下面被注释掉的thin_image函数（待改）
--------------------------------------------------------------------------------------*/

//-----------------------------头文件声明部分-----------------------------------------//
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<string>
#include <algorithm>

//----------------------------命名空间------------------------------------------------//
using namespace cv;
using namespace std;

//---------------------------函数声明部分--------------------------------------------//
double getThreshVal_Otsu_8u(const Mat& _src);
void bwareaopen(Mat& data, int n);
void ls_LED(const Mat& _img, int& X_min, int& X_max, int& Y_min, int& Y_max, Mat& imgNext, int ii);
//Mat thin_image(Mat& imgCut1, Mat& thin_img_cut);

//--------------------------main函数部分---------------------------------------------//
int main()
{
	Mat imageLED1;
	imageLED1 = imread("image1.jpg", 1);
	resize(imageLED1, imageLED1, Size(1280, 960), 0, 0, INTER_NEAREST);
	//imshow("【原图】", imageLED1);

	Mat grayImage;
	cvtColor(imageLED1, grayImage, COLOR_BGR2GRAY);
	//imshow("grayImage", grayImage);

	double m_threshold;
	Mat matBinary;
	m_threshold = getThreshVal_Otsu_8u(grayImage);
	threshold(grayImage, matBinary, m_threshold, 255, 0);
	//imshow("matBinary", matBinary);

	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
	morphologyEx(matBinary, matBinary, MORPH_CLOSE, element);
	//imshow("闭运算效果", matBinary);

	int Img_local_X1, Img_local_Y1, Img_local_X2, Img_local_Y2, Img_local_X3, Img_local_Y3;
	Mat img1_next, matBinary11, img2_next, matBinary2, img3_next, matBinary3;
	int X1_min, X1_max, Y1_min, Y1_max, X2_min, X2_max, Y2_min, Y2_max, X3_min, X3_max, Y3_min, Y3_max;

	for (int ii = 1; ii < 4; ii++)
	{
		int X_min, X_max, Y_min, Y_max;
		Mat img_next;
		ls_LED(matBinary, X_min, X_max, Y_min, Y_max, img_next, ii);                                 //获得一个ROI

		double Img_local_X = (X_max + X_min) / 2;                                                //获得LED1像素中心的位置
		double Img_local_Y = (Y_max + Y_min) / 2;

		//将原图中LED1部分的区域变黑(变黑的目的是为了下一次使用1s_LED的时候可以检测到其他ROI)

		// 获取图像的行列
		double rowB = matBinary.rows;															 // 二值化图像的行数
		double colB = matBinary.cols;															 //二值化图像的列数
		Mat matBinary1 = matBinary.clone();
		/*
		for (double i = 0;i < rowB;i++)
		{
			for (double j = 0;j < colB;j++)
			{
				double r = pow((i - Img_local_Y), 2) + pow((j - Img_local_X), 2) - pow(((abs(X_max - X_min)) / 2 - 2), double(2));
				if (r - 360 > 0)//
				{
					//
					matBinary1.at<uchar>(i, j) = matBinary.at<uchar>(i, j);
				}
				else
				{
					matBinary1.at<uchar>(i, j) = 0;//
				}
			}
		}
		matBinary = matBinary1.clone();
		//bwareaopen(matBinary, 500);//
		*/
		//----------------important--------------//
		for (int i = Y_min - 5; i < Y_max + 5; i++)
		{
			for (int j = X_min - 5; j < X_max + 5; j++)
			{
				matBinary.at<uchar>(i, j) = 0;
			}
		}


		//-------------下面看一下每一次ls_LED之后也没有把已经检测出来的区域变黑------------//
		/*
		stringstream temp;
		temp<<ii;
		string s1 = temp.str();
		imshow(s1, matBinary);
		*/

		switch (ii)                                                            //ii表示是哪个灯
		{
		case 1:
			img1_next = img_next.clone();
			Img_local_X1 = Img_local_X;
			Img_local_Y1 = Img_local_Y;
			matBinary11 = matBinary1.clone();

			X1_min = X_min;
			X1_max = X_max;
			Y1_min = Y_min;
			Y1_max = Y_max;
			break;
		case 2:
			img2_next = img_next.clone();
			Img_local_X2 = Img_local_X;
			Img_local_Y2 = Img_local_Y;
			matBinary2 = matBinary1.clone();

			X2_min = X_min;
			X2_max = X_max;
			Y2_min = Y_min;
			Y2_max = Y_max;
			break;
		case 3:
			img3_next = img_next.clone();
			Img_local_X3 = Img_local_X;
			Img_local_Y3 = Img_local_Y;
			matBinary3 = matBinary1.clone();

			X3_min = X_min;
			X3_max = X_max;
			Y3_min = Y_min;
			Y3_max = Y_max;
			break;
		}
	}

	Mat imageLED = imageLED1(Rect(X1_min, Y1_min, X1_max - X1_min, Y1_max - Y1_min));
	//imageLED = thin_image(imageLED, imageLED);
	//imshow("select_ROI_1", imageLED);                                                                    //输出对应的ROI区域  

	Mat imageLED2 = imageLED1(Rect(X2_min, Y2_min, X2_max - X2_min, Y2_max - Y2_min));
	//imageLED2 = thin_image(imageLED2, imageLED2);
	//imshow("select_ROI_2", imageLED2);                                                                    //输出对应的ROI区域  

	Mat imageLED3 = imageLED1(Rect(X3_min, Y3_min, X3_max - X3_min, Y3_max - Y3_min));
	//imageLED3 = thin_image(imageLED3, imageLED3);
	imshow("select_ROI_3", imageLED3);                                                                    //输出对应的ROI区域  

	cvtColor(imageLED, imageLED, COLOR_BGR2GRAY);

	Mat msgDateoringal = imageLED3.col(imageLED3.size().height / 2);                                       //msgDateoringal表示中间列像素矩阵
	//cout<<msgDateoringal.t().size()<<endl;                                                            一个LED_ROI的维度是 46行1列

	cout << "-----------------------------------------" << endl;
	cout << "中间列像素msgDate = " << msgDateoringal.t() << endl;                                   //将消息输出出来

	int backgroundThreshold = 20;                                                                        //设置20为阈值
	Mat maskOfimgLED;
	threshold(imageLED, maskOfimgLED, backgroundThreshold, 1, THRESH_BINARY);
	// 取阈值以上值的均值，逻辑是运用掩模，其中的数值为0或者1，为1的地方，计算出image中所有元素的均值，为0的地方，不计算


	Mat msgDate = imageLED.col(0).t();
	int meanOfPxielRow;                                                                                //.val[0]表示第一个通道的均值
	MatIterator_<uchar> it, end;

	int RowOfimgLED = 0;

	for (it = msgDate.begin<uchar>(), end = msgDate.end<uchar>(); it != end; it++) {
		meanOfPxielRow = mean(imageLED.row(RowOfimgLED), maskOfimgLED.row(RowOfimgLED)).val[0];
		RowOfimgLED++;
		// cout << "值 = "<< meanOfPxielRow <<std::endl;
		*it = meanOfPxielRow;
	}
	cout << "-----------------------------------------" << endl;
	cout << "插值前 = " << msgDate << endl;

	msgDate = msgDate.t();

	//---------------------------------------------下面对信号做插值处理------------------------------------------//
	Mat msgDate_resize;

	// cout << "size:" << msgDate.size() << endl;
	// cout << "row:" << msgDate.rows << endl;
	// cout << "col:" << msgDate.cols << endl;

	double chazhi = 3.9;                                                                                 //大小变了，对应插值变但采样不变
	resize(msgDate, msgDate_resize, Size(1, msgDate.rows * chazhi), INTER_CUBIC);

	cout << "-----------------------------------------" << endl;
	cout << "插值后信号数目 = " << msgDate_resize.rows << endl;

	cout << "-----------------------------------------" << endl;
	cout << "插值msgDate_resize= " << msgDate_resize.t() << endl;                                        //将插值后的输出出来,msgDate_resize这个是对列矩阵线性插值之后的结果
	//cout << "123456= "<< msgDate_resize.size() <<endl;


	vector<Point> in_point;

	//-------------关于in_point的几点说明-----------------//
	//1.in_point是一个vector容器,里面的每一个元素就是一个Point类型
	//2.每一个Point类型的元素是(x, y), 但注意此时不表示二维的坐标
	//x-->表示这个单列的像素矩阵的某一个像素对应的行数
	//y-->表示这个单列的像素矩阵x位置处的像素值


	for (int i = 0; i <= msgDate_resize.rows - 1; i++)                                                        //插值之后一共是179个像素
	{
		int y = msgDate_resize.at<uchar>(i, 0);                                                          //像素值

		in_point.push_back(Point(i, y));
	}
	cout << "-----------------------------------------" << endl;
	cout << "in_Point容器的大小:" << in_point.size() << endl;


	//-----------------------------------------------下面实现EVA算法-----------------------------------------//
	double minVal, maxVal;                                                                                //定义最大与最小的像素
	int minIdx[2] = {}, maxIdx[2] = {};																      //最大值、最小值对应的坐标

	minMaxIdx(msgDate_resize, &minVal, &maxVal, minIdx, maxIdx);//这个可以求最大最小值值

	cout << "-----------------------------------------" << endl;
	cout << "最大值:   " << minVal << endl;
	cout << "最小值:   " << maxVal << endl;
	cout << "最小值对应的真实二维坐标:   (" << minIdx[1] << " ," << minIdx[0] << ")" << endl;                        //因此,映射到in_Point中,位置坐标我们需要选取的是Idx[0]
	cout << "最大值对应的真实二维坐标:   (" << maxIdx[1] << " ," << maxIdx[0] << ")" << endl;


	cout << "-----------------------------------------" << endl;
	cout << "最大值与in_Point的映射关系:" << in_point[maxIdx[0]].y << endl;

	vector<Point> local_maxmin_threshold;																//（x,y）x就是第几个像素，y就是对应像的阈值。类似in_point
	int Flag_minmax = 2;																			    //用于奇偶判断:偶数表示需要找极小值;奇数表示需要找极大值

	double maxminVal = maxVal;                                                                          //初始化是最大值,即论文里面的G_max  
	int next_point = 0;

	for (int i = maxIdx[0]; i <= msgDate_resize.rows; i = i + next_point)                                 //先从最大值开始往数组的右边找极值
	{
		//第一遍循环的时候,i是最大值所对应的位置maxIdx[0]
		double value = 0;                                                                                 //存放下一个极值
		double average_gobal_maxmin;																	//存放阈值(每次更新)

		if (Flag_minmax % 2 == 0)                                                                       //若为偶数,就表示需要往右边找的是极小值
		{
			double minVal1, maxVal1;															        //最大与最小的像素
			int minIdx1[2] = {}, maxIdx1[2] = {};														//极值对应的坐标。由于是向量，以0为索引就好了

			Mat maxmin_ROI = msgDate_resize(Rect(0, i + 9, 1, 9));                                        // Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
			//cout<<"x"<<in_point[53].y;

			//-----------------关于为什么是 i+9的解读-------------//
			//首先由于我们在本代码中设定的采样率是9 pixels/bit,那么根据论文所描述的
			//为了防止误把噪声算进去,那么下一次判断的像素距离上一个极值应该至少大于
			//这个最小采样间隔,也就是9
			//---------------------------------------------------//

			cout << "-----------------------------------------" << endl;
			cout << "********************现在执行的是极小值的搜索 (向右搜索过程)************************" << endl;
			cout << "搜索范围是: " << maxmin_ROI;

			minMaxIdx(maxmin_ROI, &minVal1, &maxVal1, minIdx1, maxIdx1);                                //寻找这个区域内的最大与最小值
			//NOTE: !! 因为我是在maxmin_ROI里面去找的最小值，所以这个最小值的坐标是相对于maxmin_ROI的，而不是相对于原图的

			//value=in_point[minIdx1[0]].y;                                                               //当前值为最小值，将其赋予value

			value = minVal1;

			cout << "-----------------------------------------" << endl;

			cout << "我们找到的极小值是:" << value << endl;
			cout << "其相对坐标是:" << minIdx1[0] << endl;

			average_gobal_maxmin = (maxminVal + value) / 2;												    //计算区域阈值

			next_point = minIdx1[0] + 9;																	//minIdx1[0]为（i+9, i+2*9)这个区域的最小值坐标。

			//----------------------------------------这一个for循环用于确定阈值的作用范围--------------------------------------------------//
			for (int j = i + 1, index = 0; j <= i + next_point; j++, index++)//对比的时候是从（i+9,i+2*9）这个区域找，但实际上，是上一个极值到当前的极值的位置
			{
				local_maxmin_threshold.push_back(Point(j, average_gobal_maxmin));
				//cout<<"此时的阈值为:"<<local_maxmin_threshold.at(index)<<endl;
				//cout<<"阈值数组的大小为"<<local_maxmin_threshold.size()<<endl;
			}
			Flag_minmax++;
			maxminVal = value;
		}
		else                                                                                            //若为奇数就是求极大值了
		{
			double minVal1, maxVal1;																	//最大与最小的像素
			int minIdx1[2] = {}, maxIdx1[2] = {};														//对应的坐标。

			Mat maxmin_ROI = msgDate_resize(Rect(0, i + 9, 1, 9));											// Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);

			minMaxIdx(maxmin_ROI, &minVal1, &maxVal1, minIdx1, maxIdx1);								//寻找这个区域内的最大与最小值
			cout << "-----------------------------------------" << endl;
			cout << "************************现在执行的是极大值的搜索 (向右搜索过程)*****************************" << endl;
			cout << "搜索范围是: " << maxmin_ROI << endl;
			//value=in_point[maxIdx1[0]].y;																//当前值为最大值，将其赋予value
			value = maxVal1;


			cout << "-----------------------------------------" << endl;

			//cout<<"当前的极小值为:"<<maxminVal<<endl;
			cout << "我们找到的的极大值为:" << value << endl;
			cout << "相对坐标是:" << maxIdx1[0] << endl;

			average_gobal_maxmin = (maxminVal + value) / 2;

			next_point = maxIdx1[0] + 9;																	//minIdx1[0]为（i+9, i+2*9)这个区域的最大值坐标。

			//cout<<"目前的位置:"<<next_point;
			/*
			//----------------------------//
			cout<<"-----------------------------------------"<<endl;
			cout<<"理论上这个ROI的最大值是:"<<maxVal1<<endl;
			cout<<"理论上这个ROI最小值的索引应该是:"<<maxIdx1[0]<<endl;
			cout<<"理论上这个索引对应的in_point的坐标是:"<<in_point[maxIdx[0]+ next_point]<<endl;
			cout<<"实际上我们找到的最大值是:"<<value<<endl;
			*/


			for (int j = i; j <= i + next_point; j++)                                                           //对比的时候是从（i+9,i+2*9）这个区域找，但实际上，是上一个极值到当前的极值的位置
			{
				local_maxmin_threshold.push_back(Point(j, average_gobal_maxmin));
			}

			Flag_minmax++;																			   //用于奇偶判断，此循环是奇数才进行，加了后变成偶数，进入求极小值的循环
			// cout<<"计算极小值"<<endl;
			// cout<<"对应的极大值"<<maxminVal<<endl;
			// cout<<"value="<<value<<endl;
			maxminVal = value;//求完极大值，将当前的极大值赋值，用于求下一个极小值
		}
		if (i + next_point + 2 * 9 >= msgDate_resize.rows)                                                     //当前的点是否已经不支持下一次判决，一判决就会溢出.那么就将当前的阈值赋上
		{
			double minVal1, maxVal1;																   //最大与最小的像素
			int minIdx1[2] = {}, maxIdx1[2] = {};	                                                   //对应的坐标

			Mat maxmin_ROI = msgDate_resize(Rect(0, i + 9, 1, msgDate_resize.rows - (i + 9)));                 //从剩下的位置里面找一下极值

			minMaxIdx(maxmin_ROI, &minVal1, &maxVal1, minIdx1, maxIdx1);							   //寻找这个区域内的最大与最小值
			if (Flag_minmax % 2 == 0)																   //偶数则极小
			{
				// value=in_point[minIdx1[0]].y;														   //当前值为最小值，将其赋予value
				value = minVal1;
			}
			else
			{
				//value=in_point[maxIdx1[0]].y;														   //当前值为最大值，将其赋予value
				value = maxVal1;
			}
			average_gobal_maxmin = (maxminVal + value) / 2;
			for (int j = i; j <= msgDate_resize.rows; j++)
			{
				local_maxmin_threshold.push_back(Point(j, average_gobal_maxmin));
			}
			break;//跳出循环
		}
	}
	//注意上面这个大的for循环只是完成了在最大值以下的搜索，最大值以上的搜索还需要额外的一个循环

	Flag_minmax = 2;																					  //用于奇偶判断
	maxminVal = maxVal;																				  //初始化是最大值
	next_point = 0;

	//---------------------------现在是以最大值为界限向左搜索---------------------------//
	for (int i = maxIdx[0]; i >= 0; i = i - next_point)                                                         //由于是极值点上一个开始寻找
	{
		double value = 0;																				  //存放下一个极值
		double average_gobal_maxmin;																  //存放阈值(每次更新)

		// cout<<"i="<<i<<endl;
		// cout<<"next_point="<<next_point<<endl;
		// cout<<"Flag_minmax="<<Flag_minmax<<endl;

		if (Flag_minmax % 2 == 0)																	  //若为偶数就寻找极小值
		{
			double minVal1, maxVal1;															      //最大与最小的像素
			int minIdx1[2] = {}, maxIdx1[2] = {};	                                                  //对应的坐标

			Mat maxmin_ROI = msgDate_resize(Rect(0, i - 2 * 9, 1, 9));								      // Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
			minMaxIdx(maxmin_ROI, &minVal1, &maxVal1, minIdx1, maxIdx1);							  //寻找这个区域内的最大与最小值

			cout << "-----------------------------------------" << endl;
			cout << "************************现在执行的是极小值的搜索 (向左搜索过程)*****************************" << endl;
			cout << "-----------------------------------------" << endl;
			cout << "搜索范围是: " << maxmin_ROI << endl;
			// value=in_point[minIdx1[0]].y;															  //当前值为最小值，将其赋予value
			value = minVal1;

			cout << "-----------------------------------------" << endl;
			cout << "我们找到的极小值是:" << value << endl;
			cout << "相对位置是:" << minIdx1[0] << endl;

			average_gobal_maxmin = (maxminVal + value) / 2;

			next_point = 9 - minIdx1[0];																  //minIdx1[0]为（i+9, i+2*9)这个区域的最小值坐标。

			//-----------------确定阈值的作用范围----------------//
			for (int j = i; j >= i - next_point; j--)//对比的时候是从（i+9,i+2*9）这个区域找，但实际上，是上一个极值到当前的极值的位置
			{
				local_maxmin_threshold.insert(local_maxmin_threshold.begin(), Point(j, average_gobal_maxmin));
			}
			Flag_minmax++;																			 //用于奇偶判断，此循环是偶数才进行，加了后变成奇数，进入求极大值的循环
			maxminVal = value;																		 //求完极小值，将当前的极小值赋值，用于求下一个阈值
		}
		else
		{
			double minVal1, maxVal1;																//最大与最小的像素
			int minIdx1[2] = {}, maxIdx1[2] = {};

			Mat maxmin_ROI = msgDate_resize(Rect(0, i - 2 * 9, 1, 9));
			minMaxIdx(maxmin_ROI, &minVal1, &maxVal1, minIdx1, maxIdx1);					        //寻找这个区域内的最大与最小值

			cout << "-----------------------------------------" << endl;
			cout << "************************现在执行的是极大值的搜索 (向左搜索过程)*****************************" << endl;
			cout << "-----------------------------------------" << endl;

			//value=in_point[maxIdx1[0]].y;															//当前值为最大值，将其赋予value
			value = maxVal1;

			cout << "搜索范围是: " << maxmin_ROI << endl;
			cout << "-----------------------------------------" << endl;
			cout << "我们找到的极大值是:" << value << endl;
			cout << "相对位置是:" << maxIdx1[0] << endl;

			average_gobal_maxmin = (maxminVal + value) / 2;

			next_point = 9 - maxIdx1[0];														        //minIdx1[0]为（i+9, i+2*9)这个区域的最大值坐标。

			//-----------------确定阈值的作用范围----------------//
			for (int j = i; j >= i - next_point; j--)//对比的时候是从（i+9,i+2*9）这个区域找，但实际上，是上一个极值到当前的极值的位置
			{
				local_maxmin_threshold.insert(local_maxmin_threshold.begin(), Point(j, average_gobal_maxmin));
			}

			Flag_minmax++;
			maxminVal = value;																		//求完极大值，将当前的极大值赋值，用于求下一个阈值
		}
		//-------------最后的情况------------//			
		if (i - next_point - 2 * 9 <= 0)																	//当前的点是否已经不支持下一次判决，一判决就会溢出.那么就将当前的阈值赋上
		{
			double minVal1, maxVal1;												                //最大与最小的像素
			int minIdx1[2] = {}, maxIdx1[2] = {};

			Mat maxmin_ROI = msgDate_resize(Rect(0, 0, 1, 18));
			minMaxIdx(maxmin_ROI, &minVal1, &maxVal1, minIdx1, maxIdx1);							//寻找这个区域内的最大与最小值

			if (Flag_minmax % 2 == 0)																//偶数则极小
			{
				//value=in_point[minIdx1[0]].y;														//当前值为最小值，将其赋予value
				value = minVal1;
			}
			else
			{
				value = in_point[maxIdx1[0]].y;														//当前值为最小值，将其赋予value
				value = maxVal1;
			}
			average_gobal_maxmin = (maxminVal + value) / 2;
			for (int j = i; j >= 0; j--)
			{
				local_maxmin_threshold.insert(local_maxmin_threshold.begin(), Point(j, average_gobal_maxmin));
			}
			break;//跳出循环
		}
	}
	cout << "--------------------------------------------------------------" << endl;
	cout << "EVA的阈值结果" << local_maxmin_threshold << endl;


	int sample_point = 0;																					//采样点（0～9）
	int sample_interval = 9;																				//采样间隔


sample_again: std::cout << "******sample_again   " << sample_point << "次" << endl;

	vector<int> BitVector;
	vector<int> local_maxmin_;

	double pxielFlag;
	for (int i = sample_point; i <= msgDate_resize.rows; i = i + sample_interval)
	{
		BitVector.push_back(msgDate_resize.at<uchar>(i));                                               //数据采样--
		local_maxmin_.push_back(local_maxmin_threshold[i].y);                                           //EVA阈值采样局部自适应阈值采样	
	}

	cout << "对数据进行采样的结果是: " << Mat(BitVector, true).t() << endl;								//将采样后的输出出来
	cout << "对EVA阈值进行采样的结果是:" << Mat(local_maxmin_, true).t() << endl;

	vector<int> BitVector_local_maxmin_ = BitVector;
	cout << BitVector.size();
	for (int i = 0; i != BitVector.size(); i++)
	{

		if (BitVector_local_maxmin_[i] <= local_maxmin_[i])
		{
			BitVector_local_maxmin_[i] = 0;
		}
		else
		{
			BitVector_local_maxmin_[i] = 1;
		}

	}
	cout << "--------------------------------------------------------------" << endl;
	cout << "EVA解码的结果是: " << Mat(BitVector_local_maxmin_, true).t() << endl;



	//--------------------------------WARNING : 下面这一段代码内含 BUG ，如果仅仅是想测试上面EVA算法就把这里注释掉！-----------------------------//
	/*
	///////////////////////////////////////////////////////////////////////////////
	Mat msgDataVector;
	msgDataVector=Mat(BitVector_local_maxmin_, true).t();

	msgDataVector.convertTo(msgDataVector, CV_8U);

	Mat Header = (cv::Mat_<uchar>(1, 5) <<  1, 0, 1, 0, 1);
	Mat result(msgDataVector.rows - Header.rows + 1, msgDataVector.cols-Header.cols + 1, CV_8U);//创建模板匹配法输出结果的矩阵
	matchTemplate(msgDataVector, Header, result, CV_TM_CCOEFF_NORMED);

	threshold(result, result, 0.8, 1., CV_THRESH_TOZERO);

	vector<int> HeaderStamp;//存放消息头的位置

	while (true) {
		double minval, maxval, threshold = 0.8;
		cv::Point minloc, maxloc;
		cv::minMaxLoc(result, &minval, &maxval, &minloc, &maxloc);

		if (maxval >= threshold) {
			HeaderStamp.push_back(maxloc.x);
			// 漫水填充已经识别到的区域
			cv::floodFill(result, maxloc, cv::Scalar(0), 0, cv::Scalar(.1), cv::Scalar(1.));
		} else {
			break;
		}
	}


	int which_threshold = 4;

	// 在两个消息头之间提取ROI区域，即位ID信息
	int ptrHeaderStamp = 0;
	cv::Mat LED_ID;
	getROI:
	try {
		LED_ID=msgDataVector.colRange(HeaderStamp.at(ptrHeaderStamp) + Header.size().width,
							HeaderStamp.at(ptrHeaderStamp + 1));
		//colRange（start，end），包含的范围是不保护start列，包含end列

	} catch ( cv::Exception& e ) {  // 异常处理
		ptrHeaderStamp++;
		// const char* err_msg = e.what();
		// std::cout << "exception caught: " << err_msg << std::endl;
		std::cout << "正常现象，切勿惊慌" << std::endl;
		goto getROI;
	} catch ( std::out_of_range& e ) {  // 异常处理
		std::cout << "此LED图像ID无法识别" << std::endl;
		std::cout << "sample_point="<<sample_point << std::endl;
		if (which_threshold==0)//采用第一种方法
		{
			sample_point++;
			if (sample_point<=sample_interval)
			{
				goto sample_again;//重新采样
			}
			sample_point=-1;//由于下面循环先进入++，而采样范围为0～9
			which_threshold++;
		}
		if (which_threshold==1)//采用第二种方法
		{

			sample_point++;
			if (sample_point<=sample_interval)
			{
				goto sample_again;
			}
			sample_point=-1;
			which_threshold++;
		}
		if (which_threshold==2)
		{
			sample_point++;
			if (sample_point<=sample_interval)
			{
				goto sample_again;
			}
			sample_point=-1;
			which_threshold++;
		}
		if (which_threshold==3)
		{
			sample_point++;
			if (sample_point<=sample_interval)
			{
				goto sample_again;
			}
			sample_point=-1;
			which_threshold++;
		}
		if (which_threshold==4)
		{
			sample_point++;
			if (sample_point<=sample_interval)
			{
				goto sample_again;
			}
			sample_point=-1;
			which_threshold++;
		}
	}

	std::cout << "LED_ID="<<LED_ID << std::endl;
	// std::cout << "which_threshold="<<which_threshold << std::endl;

	switch (which_threshold)
	{
		case 0:
		std::cout << "自适应阈值判断成功" << std::endl;
		break;
		case 1:
		std::cout << "多项式判断成功" << std::endl;
		break;
		case 2:
		std::cout << "小区域自适应阈值判断成功" << std::endl;
		break;
		case 3:
		std::cout << "局部自适应阈值判断成功" << std::endl;
		break;
		case 4:
		std::cout << "EA阈值判断成功" << std::endl;
		break;
	}

	*/

	waitKey();
	return 0;
}





//------------[Funtion to calculate the threshold----------//
double getThreshVal_Otsu_8u(const Mat& _src)
{
	Size size = _src.size();
	if (_src.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
	}
	const int N = 256;
	int i, j, h[N] = { 0 };
	for (i = 0; i < size.height; i++)
	{
		const uchar* src = _src.data + _src.step * i;
		for (j = 0; j <= size.width - 4; j += 4)
		{
			int v0 = src[j], v1 = src[j + 1];
			h[v0]++; h[v1]++;
			v0 = src[j + 2]; v1 = src[j + 3];
			h[v0]++; h[v1]++;
		}
		for (; j < size.width; j++)
			h[src[j]]++;
	}

	double mu = 0, scale = 1. / (size.width * size.height);
	for (i = 0; i < N; i++)
		mu += i * h[i];

	mu *= scale;
	double mu1 = 0, q1 = 0;
	double max_sigma = 0, max_val = 0;

	for (i = 0; i < N; i++)
	{
		double p_i, q2, mu2, sigma;

		p_i = h[i] * scale;
		mu1 *= q1;
		q1 += p_i;
		q2 = 1. - q1;

		if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON)
			continue;

		mu1 = (mu1 + i * p_i) / q1;
		mu2 = (mu - q1 * mu1) / q2;
		sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
		if (sigma > max_sigma)
		{
			max_sigma = sigma;
			max_val = i;
		}
	}

	return max_val;
}

/*
void bwareaopen(Mat &data, int n)
{
	Mat labels, stats, centroids;
	connectedComponentsWithStats(data, labels, stats, centroids, 8, CV_16U);
	int regions_count = stats.rows - 1;
	int regions_size, regions_x1, regions_y1, regions_x2, regions_y2;

	for (int i = 1;i <= regions_count;i++)
	{
		regions_size = stats.ptr<int>(i)[4];
		if (regions_size < n)
		{
			regions_x1 = stats.ptr<int>(i)[0];
			regions_y1 = stats.ptr<int>(i)[1];
			regions_x2 = regions_x1 + stats.ptr<int>(i)[2];
			regions_y2 = regions_y1 + stats.ptr<int>(i)[3];

			for (int j = regions_y1;j<regions_y2;j++)
			{
				for (int k = regions_x1;k<regions_x2;k++)
				{
					if (labels.ptr<ushort>(j)[k] == i)
						data.ptr<uchar>(j)[k] = 0;
				}
			}
		}
	}
}

*/

//-------------------------------[Function for ROI detection]--------------------------------------//
void ls_LED(const Mat& _img, int& X_min, int& X_max, int& Y_min, int& Y_max, Mat& imgNext, int ii)
{
	/*
	函数说明:
	[功能]:用于获取图像里面的LED_ROI,且达到较高的鲁棒性

	[参数说明]:
	参数1: 输入的图片
	参数2: ROI的左边界
	参数3: ROI的右边界
	参数4: ROI的上边界
	参数5: ROI的下边界
	参数6: 截取到的ROI
	*/

	Mat temp1 = _img.clone();

	// 求xmin与xmax
	int row1 = temp1.rows;                               // 行数
	int col1 = temp1.cols;                               // 列
	int j = 0;                                           // 注意是从0开始
	while (j < col1)
	{
		double sum1 = 0.0;
		for (int i = 0; i < row1; i++)      				 // 注意没有等于号
		{
			uchar* data1 = temp1.ptr<uchar>(i);          // ptr<uchar>(i)[j]访问第i行第j列的像素
			sum1 = data1[j] + sum1;
		}                                                // 将第j列的每一行加完
		if (sum1 > -0.000001 && sum1 < 0.000001)            // double类型，不能写==0,因此满足这个条件就可以视作等于sum=0
		{
			j++;                                         //如果整一列都没有像素值,那么必然是没有包括LED的ROI的
		}
		else
		{
			break;										 // 跳出这个while循环
		}

	}
	X_min = j;                                           //X_min表示图像里面最早出现LED部分的列数                                            

	while (j < col1)                                     // j的初值为X_min
	{
		double sum1 = 0.0;
		for (int i = 0; i < row1; i++)
		{
			uchar* data1 = temp1.ptr<uchar>(i);          // ptr<uchar>(i)[j]访问第i行第j列的像素
			sum1 = data1[j] + sum1;
		}                                                // 将第j列的每一行加完
		if (sum1 != 0)
		{
			j++;
		}
		else
		{
			break;										// 跳出这个while循环
		}
	}
	X_max = j;                                          //这里X_max表示这个LED的ROI最右边的边界

	// 进行切割
	Mat imgCut = temp1(Rect(X_min, 0, X_max - X_min, row1));  //先把包含这个ROI的整一条全部取出来先

	/*
	stringstream temp_index;
	temp_index << ii+10;
	string s2 = temp_index.str();
	imshow(s2, imgCut);
	*/

	Mat temp = imgCut.clone();
	//-------------------以上步骤我们只是获取了ROI的左右边界,下面开始获取上下边界--------------//

	// 求ymin与ymax
	int row = temp.rows;							   // 行数
	int col = temp.cols;							   // 列
	int i = 0;
	while (i < row)                                    // i的初值为1
	{
		double sum = 0.0;
		uchar* data = temp.ptr<uchar>(i);
		for (j = 0; j < col; j++)                        // 对每一行中的每一列像素进行相加，ptr<uchar>(i)[j]访问第i行第j列的像素
		{
			sum = data[j] + sum;
		}                                              // 最终获得第i行的列和
		if (sum > -0.000001 && sum < 0.000001)
		{
			i++;                                       //如果整一行的像素和都是0,那这一行必定是没有ROI区域的
		}
		else
		{
			Y_min = i;                                 //否则,那么说明这一行就是我们找到的ROI上边界
			break;								       // 跳出这个while循环
		}
	}
	Y_min = i;

	while (i <= row - 16)                              // i的初值为Y_min   #####-16
	{
		double sum = 0.0;
		uchar* data = temp.ptr<uchar>(i);
		for (j = 0; j < col; j++)                        // 对每一行中的每一列像素进行相加，ptr<uchar>(i)[j]访问第i行第j列的像素
		{
			sum = data[j] + sum;
		}											   // 最终获得第i行的列和
		if (sum != 0)
		{
			i++;									   //sum!=0说明还没有到ROI的下边界
		}
		else
		{
			//--------------------下面的步骤用于防止有没有提前结束---------------------//
			/*因为对于发射0信号的某一行而言,经过二值处理之后,去检测它这一行的像素总和也是0
			单这并不意味着LED的ROI就到此结束了,我们还需要往后多检测几行看看每行还是不是0
			*/

			double sum6 = 0.0;
			int iiii = i + 16;
			uchar* data = temp.ptr<uchar>(iiii);
			for (j = 0; j < col; j++)					   // 对每一行中的每一列像素进行相加，ptr<uchar>(i)[j]访问第i行第j列的像素
			{
				sum6 = data[j] + sum6;
			}										   // 最终获得第i行之后20行，即iiii的列和
			if (sum6 > -0.000001 && sum6 < 0.000001)   // 如果仍然为0，才跳出
			{
				Y_max = i;
				goto logo;							   // 跳出这个while循环
			}
			else// 否则继续执行
			{
				i++;
			}
		}
	}
logo:
	Y_max = i;

	// 进行切割
	Mat imgCut1 = temp(Rect(0, Y_min, col, Y_max - Y_min));
	imgNext = imgCut1.clone();						  // clone函数创建新的图片
	/*
	//---------------对于Cut之后的ROI进行最后一次检查-----------------//
	int j_last= 0;
	int row_last = imgCut1.rows;
	int col_last = imgCut1.cols;
	while(j_last < col_last)
	{
		double sum_last = 0.0;    //定义列像素总和为0
		for (int i_last = 0;i_last < row_last; i_last ++)      				 // 注意没有等于号
		{
			uchar* data2 = imgCut1.ptr<uchar>(i_last);          // ptr<uchar>(i)[j]访问第i行第j列的像素
			sum_last = data2[j_last] + sum_last;
		}                                                // 将第j列的每一行加完
		if (sum_last>-0.000001 && sum_last< 0.000001)            // double类型，不能写==0,因此满足这个条件就可以视作等于sum=0
		{
			j_last++;                                         //如果整一列都没有像素值,那么必然是没有包括LED的ROI的
		}
		else
		{
			break;										 // 跳出这个while循环
		}
	}
	X_min = j_last;

	while (j_last < col_last)                                     // j的初值为X_min
	{
		double sum_last = 0.0;
		for (int i_last = 0;i_last < row_last;i_last++)
		{
			uchar* data2 = imgCut1.ptr<uchar>(i_last);          // ptr<uchar>(i)[j]访问第i行第j列的像素
			sum_last = data2[j_last] + sum_last;
		}                                                // 将第j列的每一行加完
		if (sum_last != 0)
		{
			j_last++;
		}
		else
		{
			break;										// 跳出这个while循环
		}
	}
	X_max = j_last;                                          //这里X_max表示这个LED的ROI最右边的边界

	Mat imgCut2 = imgCut1(Rect(X_min, 0, X_max - X_min, row_last));
	imgNext = imgCut2.clone();
	*/
}


/*
Mat thin_image(Mat& imgCut1, Mat& thin_img_cut)
{
	int X_min, X_max;
	int j_last= 0;
	int row_last = imgCut1.rows;
	int col_last = imgCut1.cols;
	while(j_last < col_last)
	{
		double sum_last = 0.0;    //定义列像素总和为0
		for (int i_last = 0;i_last < row_last; i_last ++)      				 // 注意没有等于号
		{
			uchar* data2 = imgCut1.ptr<uchar>(i_last);          // ptr<uchar>(i)[j]访问第i行第j列的像素
			sum_last = data2[j_last] + sum_last;
			cout<<"一列的像素和:"<<sum_last<<endl;
		}                                                // 将第j列的每一行加完
		if (sum_last < 800)            // double类型，不能写==0,因此满足这个条件就可以视作等于sum=0
		{
			j_last++;                                         //如果整一列都没有像素值,那么必然是没有包括LED的ROI的
		}
		else
		{
			break;										 // 跳出这个while循环
		}
	}
	X_min = j_last;

	while (j_last < col_last)                                     // j的初值为X_min
	{
		double sum_last = 0.0;
		for (int i_last = 0;i_last < row_last;i_last++)
		{
			uchar* data2 = imgCut1.ptr<uchar>(i_last);          // ptr<uchar>(i)[j]访问第i行第j列的像素
			sum_last = data2[j_last] + sum_last;
		}                                                // 将第j列的每一行加完
		if (sum_last != 0)
		{
			j_last++;
		}
		else
		{
			break;										// 跳出这个while循环
		}
	}
	X_max = j_last;                                          //这里X_max表示这个LED的ROI最右边的边界

	Mat imgCut2 = imgCut1(Rect(X_min, 0, X_max - X_min, row_last));
	thin_img_cut = imgCut2.clone();
	return thin_img_cut;
}
 
*/
