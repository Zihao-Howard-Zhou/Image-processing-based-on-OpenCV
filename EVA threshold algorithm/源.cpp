/*--------------------------------------------------------------------------------------
		   ��������ʹ��EVAʵ��LED�������߼����
������־:
����:ZZH South China University of Technology
ʱ��: 2020��6��29�� 17:34

�汾: 1.0.0

Ŀǰ�����ʵ�ֵĹ���:�����EVA�㷨,���Եõ�����֮����о����

��ʱδʵ�ֵĹ���: ����header,��һ����ȡID
����õķ���: ����header��010101����������,�ʿ��Բ����򵥵�ģ��ƥ��ȥ��010101�Ĳ���

�������BUG: ���ڵڶ���ROI����ȡ,������Ƕ��һС���ɫ����,����Ҫ˼����ôȥ��
����÷���: ���汻ע�͵���thin_image���������ģ�
--------------------------------------------------------------------------------------*/

//-----------------------------ͷ�ļ���������-----------------------------------------//
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<string>
#include <algorithm>

//----------------------------�����ռ�------------------------------------------------//
using namespace cv;
using namespace std;

//---------------------------������������--------------------------------------------//
double getThreshVal_Otsu_8u(const Mat& _src);
void bwareaopen(Mat& data, int n);
void ls_LED(const Mat& _img, int& X_min, int& X_max, int& Y_min, int& Y_max, Mat& imgNext, int ii);
//Mat thin_image(Mat& imgCut1, Mat& thin_img_cut);

//--------------------------main��������---------------------------------------------//
int main()
{
	Mat imageLED1;
	imageLED1 = imread("image1.jpg", 1);
	resize(imageLED1, imageLED1, Size(1280, 960), 0, 0, INTER_NEAREST);
	//imshow("��ԭͼ��", imageLED1);

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
	//imshow("������Ч��", matBinary);

	int Img_local_X1, Img_local_Y1, Img_local_X2, Img_local_Y2, Img_local_X3, Img_local_Y3;
	Mat img1_next, matBinary11, img2_next, matBinary2, img3_next, matBinary3;
	int X1_min, X1_max, Y1_min, Y1_max, X2_min, X2_max, Y2_min, Y2_max, X3_min, X3_max, Y3_min, Y3_max;

	for (int ii = 1; ii < 4; ii++)
	{
		int X_min, X_max, Y_min, Y_max;
		Mat img_next;
		ls_LED(matBinary, X_min, X_max, Y_min, Y_max, img_next, ii);                                 //���һ��ROI

		double Img_local_X = (X_max + X_min) / 2;                                                //���LED1�������ĵ�λ��
		double Img_local_Y = (Y_max + Y_min) / 2;

		//��ԭͼ��LED1���ֵ�������(��ڵ�Ŀ����Ϊ����һ��ʹ��1s_LED��ʱ����Լ�⵽����ROI)

		// ��ȡͼ�������
		double rowB = matBinary.rows;															 // ��ֵ��ͼ�������
		double colB = matBinary.cols;															 //��ֵ��ͼ�������
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


		//-------------���濴һ��ÿһ��ls_LED֮��Ҳû�а��Ѿ���������������------------//
		/*
		stringstream temp;
		temp<<ii;
		string s1 = temp.str();
		imshow(s1, matBinary);
		*/

		switch (ii)                                                            //ii��ʾ���ĸ���
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
	//imshow("select_ROI_1", imageLED);                                                                    //�����Ӧ��ROI����  

	Mat imageLED2 = imageLED1(Rect(X2_min, Y2_min, X2_max - X2_min, Y2_max - Y2_min));
	//imageLED2 = thin_image(imageLED2, imageLED2);
	//imshow("select_ROI_2", imageLED2);                                                                    //�����Ӧ��ROI����  

	Mat imageLED3 = imageLED1(Rect(X3_min, Y3_min, X3_max - X3_min, Y3_max - Y3_min));
	//imageLED3 = thin_image(imageLED3, imageLED3);
	imshow("select_ROI_3", imageLED3);                                                                    //�����Ӧ��ROI����  

	cvtColor(imageLED, imageLED, COLOR_BGR2GRAY);

	Mat msgDateoringal = imageLED3.col(imageLED3.size().height / 2);                                       //msgDateoringal��ʾ�м������ؾ���
	//cout<<msgDateoringal.t().size()<<endl;                                                            һ��LED_ROI��ά���� 46��1��

	cout << "-----------------------------------------" << endl;
	cout << "�м�������msgDate = " << msgDateoringal.t() << endl;                                   //����Ϣ�������

	int backgroundThreshold = 20;                                                                        //����20Ϊ��ֵ
	Mat maskOfimgLED;
	threshold(imageLED, maskOfimgLED, backgroundThreshold, 1, THRESH_BINARY);
	// ȡ��ֵ����ֵ�ľ�ֵ���߼���������ģ�����е���ֵΪ0����1��Ϊ1�ĵط��������image������Ԫ�صľ�ֵ��Ϊ0�ĵط���������


	Mat msgDate = imageLED.col(0).t();
	int meanOfPxielRow;                                                                                //.val[0]��ʾ��һ��ͨ���ľ�ֵ
	MatIterator_<uchar> it, end;

	int RowOfimgLED = 0;

	for (it = msgDate.begin<uchar>(), end = msgDate.end<uchar>(); it != end; it++) {
		meanOfPxielRow = mean(imageLED.row(RowOfimgLED), maskOfimgLED.row(RowOfimgLED)).val[0];
		RowOfimgLED++;
		// cout << "ֵ = "<< meanOfPxielRow <<std::endl;
		*it = meanOfPxielRow;
	}
	cout << "-----------------------------------------" << endl;
	cout << "��ֵǰ = " << msgDate << endl;

	msgDate = msgDate.t();

	//---------------------------------------------������ź�����ֵ����------------------------------------------//
	Mat msgDate_resize;

	// cout << "size:" << msgDate.size() << endl;
	// cout << "row:" << msgDate.rows << endl;
	// cout << "col:" << msgDate.cols << endl;

	double chazhi = 3.9;                                                                                 //��С���ˣ���Ӧ��ֵ�䵫��������
	resize(msgDate, msgDate_resize, Size(1, msgDate.rows * chazhi), INTER_CUBIC);

	cout << "-----------------------------------------" << endl;
	cout << "��ֵ���ź���Ŀ = " << msgDate_resize.rows << endl;

	cout << "-----------------------------------------" << endl;
	cout << "��ֵmsgDate_resize= " << msgDate_resize.t() << endl;                                        //����ֵ����������,msgDate_resize����Ƕ��о������Բ�ֵ֮��Ľ��
	//cout << "123456= "<< msgDate_resize.size() <<endl;


	vector<Point> in_point;

	//-------------����in_point�ļ���˵��-----------------//
	//1.in_point��һ��vector����,�����ÿһ��Ԫ�ؾ���һ��Point����
	//2.ÿһ��Point���͵�Ԫ����(x, y), ��ע���ʱ����ʾ��ά������
	//x-->��ʾ������е����ؾ����ĳһ�����ض�Ӧ������
	//y-->��ʾ������е����ؾ���xλ�ô�������ֵ


	for (int i = 0; i <= msgDate_resize.rows - 1; i++)                                                        //��ֵ֮��һ����179������
	{
		int y = msgDate_resize.at<uchar>(i, 0);                                                          //����ֵ

		in_point.push_back(Point(i, y));
	}
	cout << "-----------------------------------------" << endl;
	cout << "in_Point�����Ĵ�С:" << in_point.size() << endl;


	//-----------------------------------------------����ʵ��EVA�㷨-----------------------------------------//
	double minVal, maxVal;                                                                                //�����������С������
	int minIdx[2] = {}, maxIdx[2] = {};																      //���ֵ����Сֵ��Ӧ������

	minMaxIdx(msgDate_resize, &minVal, &maxVal, minIdx, maxIdx);//��������������Сֵֵ

	cout << "-----------------------------------------" << endl;
	cout << "���ֵ:   " << minVal << endl;
	cout << "��Сֵ:   " << maxVal << endl;
	cout << "��Сֵ��Ӧ����ʵ��ά����:   (" << minIdx[1] << " ," << minIdx[0] << ")" << endl;                        //���,ӳ�䵽in_Point��,λ������������Ҫѡȡ����Idx[0]
	cout << "���ֵ��Ӧ����ʵ��ά����:   (" << maxIdx[1] << " ," << maxIdx[0] << ")" << endl;


	cout << "-----------------------------------------" << endl;
	cout << "���ֵ��in_Point��ӳ���ϵ:" << in_point[maxIdx[0]].y << endl;

	vector<Point> local_maxmin_threshold;																//��x,y��x���ǵڼ������أ�y���Ƕ�Ӧ�����ֵ������in_point
	int Flag_minmax = 2;																			    //������ż�ж�:ż����ʾ��Ҫ�Ҽ�Сֵ;������ʾ��Ҫ�Ҽ���ֵ

	double maxminVal = maxVal;                                                                          //��ʼ�������ֵ,�����������G_max  
	int next_point = 0;

	for (int i = maxIdx[0]; i <= msgDate_resize.rows; i = i + next_point)                                 //�ȴ����ֵ��ʼ��������ұ��Ҽ�ֵ
	{
		//��һ��ѭ����ʱ��,i�����ֵ����Ӧ��λ��maxIdx[0]
		double value = 0;                                                                                 //�����һ����ֵ
		double average_gobal_maxmin;																	//�����ֵ(ÿ�θ���)

		if (Flag_minmax % 2 == 0)                                                                       //��Ϊż��,�ͱ�ʾ��Ҫ���ұ��ҵ��Ǽ�Сֵ
		{
			double minVal1, maxVal1;															        //�������С������
			int minIdx1[2] = {}, maxIdx1[2] = {};														//��ֵ��Ӧ�����ꡣ��������������0Ϊ�����ͺ���

			Mat maxmin_ROI = msgDate_resize(Rect(0, i + 9, 1, 9));                                        // Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
			//cout<<"x"<<in_point[53].y;

			//-----------------����Ϊʲô�� i+9�Ľ��-------------//
			//�������������ڱ��������趨�Ĳ�������9 pixels/bit,��ô����������������
			//Ϊ�˷�ֹ����������ȥ,��ô��һ���жϵ����ؾ�����һ����ֵӦ�����ٴ���
			//�����С�������,Ҳ����9
			//---------------------------------------------------//

			cout << "-----------------------------------------" << endl;
			cout << "********************����ִ�е��Ǽ�Сֵ������ (������������)************************" << endl;
			cout << "������Χ��: " << maxmin_ROI;

			minMaxIdx(maxmin_ROI, &minVal1, &maxVal1, minIdx1, maxIdx1);                                //Ѱ����������ڵ��������Сֵ
			//NOTE: !! ��Ϊ������maxmin_ROI����ȥ�ҵ���Сֵ�����������Сֵ�������������maxmin_ROI�ģ������������ԭͼ��

			//value=in_point[minIdx1[0]].y;                                                               //��ǰֵΪ��Сֵ�����丳��value

			value = minVal1;

			cout << "-----------------------------------------" << endl;

			cout << "�����ҵ��ļ�Сֵ��:" << value << endl;
			cout << "�����������:" << minIdx1[0] << endl;

			average_gobal_maxmin = (maxminVal + value) / 2;												    //����������ֵ

			next_point = minIdx1[0] + 9;																	//minIdx1[0]Ϊ��i+9, i+2*9)����������Сֵ���ꡣ

			//----------------------------------------��һ��forѭ������ȷ����ֵ�����÷�Χ--------------------------------------------------//
			for (int j = i + 1, index = 0; j <= i + next_point; j++, index++)//�Աȵ�ʱ���Ǵӣ�i+9,i+2*9����������ң���ʵ���ϣ�����һ����ֵ����ǰ�ļ�ֵ��λ��
			{
				local_maxmin_threshold.push_back(Point(j, average_gobal_maxmin));
				//cout<<"��ʱ����ֵΪ:"<<local_maxmin_threshold.at(index)<<endl;
				//cout<<"��ֵ����Ĵ�СΪ"<<local_maxmin_threshold.size()<<endl;
			}
			Flag_minmax++;
			maxminVal = value;
		}
		else                                                                                            //��Ϊ���������󼫴�ֵ��
		{
			double minVal1, maxVal1;																	//�������С������
			int minIdx1[2] = {}, maxIdx1[2] = {};														//��Ӧ�����ꡣ

			Mat maxmin_ROI = msgDate_resize(Rect(0, i + 9, 1, 9));											// Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);

			minMaxIdx(maxmin_ROI, &minVal1, &maxVal1, minIdx1, maxIdx1);								//Ѱ����������ڵ��������Сֵ
			cout << "-----------------------------------------" << endl;
			cout << "************************����ִ�е��Ǽ���ֵ������ (������������)*****************************" << endl;
			cout << "������Χ��: " << maxmin_ROI << endl;
			//value=in_point[maxIdx1[0]].y;																//��ǰֵΪ���ֵ�����丳��value
			value = maxVal1;


			cout << "-----------------------------------------" << endl;

			//cout<<"��ǰ�ļ�СֵΪ:"<<maxminVal<<endl;
			cout << "�����ҵ��ĵļ���ֵΪ:" << value << endl;
			cout << "���������:" << maxIdx1[0] << endl;

			average_gobal_maxmin = (maxminVal + value) / 2;

			next_point = maxIdx1[0] + 9;																	//minIdx1[0]Ϊ��i+9, i+2*9)�����������ֵ���ꡣ

			//cout<<"Ŀǰ��λ��:"<<next_point;
			/*
			//----------------------------//
			cout<<"-----------------------------------------"<<endl;
			cout<<"���������ROI�����ֵ��:"<<maxVal1<<endl;
			cout<<"���������ROI��Сֵ������Ӧ����:"<<maxIdx1[0]<<endl;
			cout<<"���������������Ӧ��in_point��������:"<<in_point[maxIdx[0]+ next_point]<<endl;
			cout<<"ʵ���������ҵ������ֵ��:"<<value<<endl;
			*/


			for (int j = i; j <= i + next_point; j++)                                                           //�Աȵ�ʱ���Ǵӣ�i+9,i+2*9����������ң���ʵ���ϣ�����һ����ֵ����ǰ�ļ�ֵ��λ��
			{
				local_maxmin_threshold.push_back(Point(j, average_gobal_maxmin));
			}

			Flag_minmax++;																			   //������ż�жϣ���ѭ���������Ž��У����˺���ż����������Сֵ��ѭ��
			// cout<<"���㼫Сֵ"<<endl;
			// cout<<"��Ӧ�ļ���ֵ"<<maxminVal<<endl;
			// cout<<"value="<<value<<endl;
			maxminVal = value;//���꼫��ֵ������ǰ�ļ���ֵ��ֵ����������һ����Сֵ
		}
		if (i + next_point + 2 * 9 >= msgDate_resize.rows)                                                     //��ǰ�ĵ��Ƿ��Ѿ���֧����һ���о���һ�о��ͻ����.��ô�ͽ���ǰ����ֵ����
		{
			double minVal1, maxVal1;																   //�������С������
			int minIdx1[2] = {}, maxIdx1[2] = {};	                                                   //��Ӧ������

			Mat maxmin_ROI = msgDate_resize(Rect(0, i + 9, 1, msgDate_resize.rows - (i + 9)));                 //��ʣ�µ�λ��������һ�¼�ֵ

			minMaxIdx(maxmin_ROI, &minVal1, &maxVal1, minIdx1, maxIdx1);							   //Ѱ����������ڵ��������Сֵ
			if (Flag_minmax % 2 == 0)																   //ż����С
			{
				// value=in_point[minIdx1[0]].y;														   //��ǰֵΪ��Сֵ�����丳��value
				value = minVal1;
			}
			else
			{
				//value=in_point[maxIdx1[0]].y;														   //��ǰֵΪ���ֵ�����丳��value
				value = maxVal1;
			}
			average_gobal_maxmin = (maxminVal + value) / 2;
			for (int j = i; j <= msgDate_resize.rows; j++)
			{
				local_maxmin_threshold.push_back(Point(j, average_gobal_maxmin));
			}
			break;//����ѭ��
		}
	}
	//ע������������forѭ��ֻ������������ֵ���µ����������ֵ���ϵ���������Ҫ�����һ��ѭ��

	Flag_minmax = 2;																					  //������ż�ж�
	maxminVal = maxVal;																				  //��ʼ�������ֵ
	next_point = 0;

	//---------------------------�����������ֵΪ������������---------------------------//
	for (int i = maxIdx[0]; i >= 0; i = i - next_point)                                                         //�����Ǽ�ֵ����һ����ʼѰ��
	{
		double value = 0;																				  //�����һ����ֵ
		double average_gobal_maxmin;																  //�����ֵ(ÿ�θ���)

		// cout<<"i="<<i<<endl;
		// cout<<"next_point="<<next_point<<endl;
		// cout<<"Flag_minmax="<<Flag_minmax<<endl;

		if (Flag_minmax % 2 == 0)																	  //��Ϊż����Ѱ�Ҽ�Сֵ
		{
			double minVal1, maxVal1;															      //�������С������
			int minIdx1[2] = {}, maxIdx1[2] = {};	                                                  //��Ӧ������

			Mat maxmin_ROI = msgDate_resize(Rect(0, i - 2 * 9, 1, 9));								      // Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
			minMaxIdx(maxmin_ROI, &minVal1, &maxVal1, minIdx1, maxIdx1);							  //Ѱ����������ڵ��������Сֵ

			cout << "-----------------------------------------" << endl;
			cout << "************************����ִ�е��Ǽ�Сֵ������ (������������)*****************************" << endl;
			cout << "-----------------------------------------" << endl;
			cout << "������Χ��: " << maxmin_ROI << endl;
			// value=in_point[minIdx1[0]].y;															  //��ǰֵΪ��Сֵ�����丳��value
			value = minVal1;

			cout << "-----------------------------------------" << endl;
			cout << "�����ҵ��ļ�Сֵ��:" << value << endl;
			cout << "���λ����:" << minIdx1[0] << endl;

			average_gobal_maxmin = (maxminVal + value) / 2;

			next_point = 9 - minIdx1[0];																  //minIdx1[0]Ϊ��i+9, i+2*9)����������Сֵ���ꡣ

			//-----------------ȷ����ֵ�����÷�Χ----------------//
			for (int j = i; j >= i - next_point; j--)//�Աȵ�ʱ���Ǵӣ�i+9,i+2*9����������ң���ʵ���ϣ�����һ����ֵ����ǰ�ļ�ֵ��λ��
			{
				local_maxmin_threshold.insert(local_maxmin_threshold.begin(), Point(j, average_gobal_maxmin));
			}
			Flag_minmax++;																			 //������ż�жϣ���ѭ����ż���Ž��У����˺��������������󼫴�ֵ��ѭ��
			maxminVal = value;																		 //���꼫Сֵ������ǰ�ļ�Сֵ��ֵ����������һ����ֵ
		}
		else
		{
			double minVal1, maxVal1;																//�������С������
			int minIdx1[2] = {}, maxIdx1[2] = {};

			Mat maxmin_ROI = msgDate_resize(Rect(0, i - 2 * 9, 1, 9));
			minMaxIdx(maxmin_ROI, &minVal1, &maxVal1, minIdx1, maxIdx1);					        //Ѱ����������ڵ��������Сֵ

			cout << "-----------------------------------------" << endl;
			cout << "************************����ִ�е��Ǽ���ֵ������ (������������)*****************************" << endl;
			cout << "-----------------------------------------" << endl;

			//value=in_point[maxIdx1[0]].y;															//��ǰֵΪ���ֵ�����丳��value
			value = maxVal1;

			cout << "������Χ��: " << maxmin_ROI << endl;
			cout << "-----------------------------------------" << endl;
			cout << "�����ҵ��ļ���ֵ��:" << value << endl;
			cout << "���λ����:" << maxIdx1[0] << endl;

			average_gobal_maxmin = (maxminVal + value) / 2;

			next_point = 9 - maxIdx1[0];														        //minIdx1[0]Ϊ��i+9, i+2*9)�����������ֵ���ꡣ

			//-----------------ȷ����ֵ�����÷�Χ----------------//
			for (int j = i; j >= i - next_point; j--)//�Աȵ�ʱ���Ǵӣ�i+9,i+2*9����������ң���ʵ���ϣ�����һ����ֵ����ǰ�ļ�ֵ��λ��
			{
				local_maxmin_threshold.insert(local_maxmin_threshold.begin(), Point(j, average_gobal_maxmin));
			}

			Flag_minmax++;
			maxminVal = value;																		//���꼫��ֵ������ǰ�ļ���ֵ��ֵ����������һ����ֵ
		}
		//-------------�������------------//			
		if (i - next_point - 2 * 9 <= 0)																	//��ǰ�ĵ��Ƿ��Ѿ���֧����һ���о���һ�о��ͻ����.��ô�ͽ���ǰ����ֵ����
		{
			double minVal1, maxVal1;												                //�������С������
			int minIdx1[2] = {}, maxIdx1[2] = {};

			Mat maxmin_ROI = msgDate_resize(Rect(0, 0, 1, 18));
			minMaxIdx(maxmin_ROI, &minVal1, &maxVal1, minIdx1, maxIdx1);							//Ѱ����������ڵ��������Сֵ

			if (Flag_minmax % 2 == 0)																//ż����С
			{
				//value=in_point[minIdx1[0]].y;														//��ǰֵΪ��Сֵ�����丳��value
				value = minVal1;
			}
			else
			{
				value = in_point[maxIdx1[0]].y;														//��ǰֵΪ��Сֵ�����丳��value
				value = maxVal1;
			}
			average_gobal_maxmin = (maxminVal + value) / 2;
			for (int j = i; j >= 0; j--)
			{
				local_maxmin_threshold.insert(local_maxmin_threshold.begin(), Point(j, average_gobal_maxmin));
			}
			break;//����ѭ��
		}
	}
	cout << "--------------------------------------------------------------" << endl;
	cout << "EVA����ֵ���" << local_maxmin_threshold << endl;


	int sample_point = 0;																					//�����㣨0��9��
	int sample_interval = 9;																				//�������


sample_again: std::cout << "******sample_again   " << sample_point << "��" << endl;

	vector<int> BitVector;
	vector<int> local_maxmin_;

	double pxielFlag;
	for (int i = sample_point; i <= msgDate_resize.rows; i = i + sample_interval)
	{
		BitVector.push_back(msgDate_resize.at<uchar>(i));                                               //���ݲ���--
		local_maxmin_.push_back(local_maxmin_threshold[i].y);                                           //EVA��ֵ�����ֲ�����Ӧ��ֵ����	
	}

	cout << "�����ݽ��в����Ľ����: " << Mat(BitVector, true).t() << endl;								//����������������
	cout << "��EVA��ֵ���в����Ľ����:" << Mat(local_maxmin_, true).t() << endl;

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
	cout << "EVA����Ľ����: " << Mat(BitVector_local_maxmin_, true).t() << endl;



	//--------------------------------WARNING : ������һ�δ����ں� BUG ��������������������EVA�㷨�Ͱ�����ע�͵���-----------------------------//
	/*
	///////////////////////////////////////////////////////////////////////////////
	Mat msgDataVector;
	msgDataVector=Mat(BitVector_local_maxmin_, true).t();

	msgDataVector.convertTo(msgDataVector, CV_8U);

	Mat Header = (cv::Mat_<uchar>(1, 5) <<  1, 0, 1, 0, 1);
	Mat result(msgDataVector.rows - Header.rows + 1, msgDataVector.cols-Header.cols + 1, CV_8U);//����ģ��ƥ�䷨�������ľ���
	matchTemplate(msgDataVector, Header, result, CV_TM_CCOEFF_NORMED);

	threshold(result, result, 0.8, 1., CV_THRESH_TOZERO);

	vector<int> HeaderStamp;//�����Ϣͷ��λ��

	while (true) {
		double minval, maxval, threshold = 0.8;
		cv::Point minloc, maxloc;
		cv::minMaxLoc(result, &minval, &maxval, &minloc, &maxloc);

		if (maxval >= threshold) {
			HeaderStamp.push_back(maxloc.x);
			// ��ˮ����Ѿ�ʶ�𵽵�����
			cv::floodFill(result, maxloc, cv::Scalar(0), 0, cv::Scalar(.1), cv::Scalar(1.));
		} else {
			break;
		}
	}


	int which_threshold = 4;

	// ��������Ϣͷ֮����ȡROI���򣬼�λID��Ϣ
	int ptrHeaderStamp = 0;
	cv::Mat LED_ID;
	getROI:
	try {
		LED_ID=msgDataVector.colRange(HeaderStamp.at(ptrHeaderStamp) + Header.size().width,
							HeaderStamp.at(ptrHeaderStamp + 1));
		//colRange��start��end���������ķ�Χ�ǲ�����start�У�����end��

	} catch ( cv::Exception& e ) {  // �쳣����
		ptrHeaderStamp++;
		// const char* err_msg = e.what();
		// std::cout << "exception caught: " << err_msg << std::endl;
		std::cout << "�����������𾪻�" << std::endl;
		goto getROI;
	} catch ( std::out_of_range& e ) {  // �쳣����
		std::cout << "��LEDͼ��ID�޷�ʶ��" << std::endl;
		std::cout << "sample_point="<<sample_point << std::endl;
		if (which_threshold==0)//���õ�һ�ַ���
		{
			sample_point++;
			if (sample_point<=sample_interval)
			{
				goto sample_again;//���²���
			}
			sample_point=-1;//��������ѭ���Ƚ���++����������ΧΪ0��9
			which_threshold++;
		}
		if (which_threshold==1)//���õڶ��ַ���
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
		std::cout << "����Ӧ��ֵ�жϳɹ�" << std::endl;
		break;
		case 1:
		std::cout << "����ʽ�жϳɹ�" << std::endl;
		break;
		case 2:
		std::cout << "С��������Ӧ��ֵ�жϳɹ�" << std::endl;
		break;
		case 3:
		std::cout << "�ֲ�����Ӧ��ֵ�жϳɹ�" << std::endl;
		break;
		case 4:
		std::cout << "EA��ֵ�жϳɹ�" << std::endl;
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
	����˵��:
	[����]:���ڻ�ȡͼ�������LED_ROI,�Ҵﵽ�ϸߵ�³����

	[����˵��]:
	����1: �����ͼƬ
	����2: ROI����߽�
	����3: ROI���ұ߽�
	����4: ROI���ϱ߽�
	����5: ROI���±߽�
	����6: ��ȡ����ROI
	*/

	Mat temp1 = _img.clone();

	// ��xmin��xmax
	int row1 = temp1.rows;                               // ����
	int col1 = temp1.cols;                               // ��
	int j = 0;                                           // ע���Ǵ�0��ʼ
	while (j < col1)
	{
		double sum1 = 0.0;
		for (int i = 0; i < row1; i++)      				 // ע��û�е��ں�
		{
			uchar* data1 = temp1.ptr<uchar>(i);          // ptr<uchar>(i)[j]���ʵ�i�е�j�е�����
			sum1 = data1[j] + sum1;
		}                                                // ����j�е�ÿһ�м���
		if (sum1 > -0.000001 && sum1 < 0.000001)            // double���ͣ�����д==0,���������������Ϳ�����������sum=0
		{
			j++;                                         //�����һ�ж�û������ֵ,��ô��Ȼ��û�а���LED��ROI��
		}
		else
		{
			break;										 // �������whileѭ��
		}

	}
	X_min = j;                                           //X_min��ʾͼ�������������LED���ֵ�����                                            

	while (j < col1)                                     // j�ĳ�ֵΪX_min
	{
		double sum1 = 0.0;
		for (int i = 0; i < row1; i++)
		{
			uchar* data1 = temp1.ptr<uchar>(i);          // ptr<uchar>(i)[j]���ʵ�i�е�j�е�����
			sum1 = data1[j] + sum1;
		}                                                // ����j�е�ÿһ�м���
		if (sum1 != 0)
		{
			j++;
		}
		else
		{
			break;										// �������whileѭ��
		}
	}
	X_max = j;                                          //����X_max��ʾ���LED��ROI���ұߵı߽�

	// �����и�
	Mat imgCut = temp1(Rect(X_min, 0, X_max - X_min, row1));  //�ȰѰ������ROI����һ��ȫ��ȡ������

	/*
	stringstream temp_index;
	temp_index << ii+10;
	string s2 = temp_index.str();
	imshow(s2, imgCut);
	*/

	Mat temp = imgCut.clone();
	//-------------------���ϲ�������ֻ�ǻ�ȡ��ROI�����ұ߽�,���濪ʼ��ȡ���±߽�--------------//

	// ��ymin��ymax
	int row = temp.rows;							   // ����
	int col = temp.cols;							   // ��
	int i = 0;
	while (i < row)                                    // i�ĳ�ֵΪ1
	{
		double sum = 0.0;
		uchar* data = temp.ptr<uchar>(i);
		for (j = 0; j < col; j++)                        // ��ÿһ���е�ÿһ�����ؽ�����ӣ�ptr<uchar>(i)[j]���ʵ�i�е�j�е�����
		{
			sum = data[j] + sum;
		}                                              // ���ջ�õ�i�е��к�
		if (sum > -0.000001 && sum < 0.000001)
		{
			i++;                                       //�����һ�е����غͶ���0,����һ�бض���û��ROI�����
		}
		else
		{
			Y_min = i;                                 //����,��ô˵����һ�о��������ҵ���ROI�ϱ߽�
			break;								       // �������whileѭ��
		}
	}
	Y_min = i;

	while (i <= row - 16)                              // i�ĳ�ֵΪY_min   #####-16
	{
		double sum = 0.0;
		uchar* data = temp.ptr<uchar>(i);
		for (j = 0; j < col; j++)                        // ��ÿһ���е�ÿһ�����ؽ�����ӣ�ptr<uchar>(i)[j]���ʵ�i�е�j�е�����
		{
			sum = data[j] + sum;
		}											   // ���ջ�õ�i�е��к�
		if (sum != 0)
		{
			i++;									   //sum!=0˵����û�е�ROI���±߽�
		}
		else
		{
			//--------------------����Ĳ������ڷ�ֹ��û����ǰ����---------------------//
			/*��Ϊ���ڷ���0�źŵ�ĳһ�ж���,������ֵ����֮��,ȥ�������һ�е������ܺ�Ҳ��0
			���Ⲣ����ζ��LED��ROI�͵��˽�����,���ǻ���Ҫ������⼸�п���ÿ�л��ǲ���0
			*/

			double sum6 = 0.0;
			int iiii = i + 16;
			uchar* data = temp.ptr<uchar>(iiii);
			for (j = 0; j < col; j++)					   // ��ÿһ���е�ÿһ�����ؽ�����ӣ�ptr<uchar>(i)[j]���ʵ�i�е�j�е�����
			{
				sum6 = data[j] + sum6;
			}										   // ���ջ�õ�i��֮��20�У���iiii���к�
			if (sum6 > -0.000001 && sum6 < 0.000001)   // �����ȻΪ0��������
			{
				Y_max = i;
				goto logo;							   // �������whileѭ��
			}
			else// �������ִ��
			{
				i++;
			}
		}
	}
logo:
	Y_max = i;

	// �����и�
	Mat imgCut1 = temp(Rect(0, Y_min, col, Y_max - Y_min));
	imgNext = imgCut1.clone();						  // clone���������µ�ͼƬ
	/*
	//---------------����Cut֮���ROI�������һ�μ��-----------------//
	int j_last= 0;
	int row_last = imgCut1.rows;
	int col_last = imgCut1.cols;
	while(j_last < col_last)
	{
		double sum_last = 0.0;    //�����������ܺ�Ϊ0
		for (int i_last = 0;i_last < row_last; i_last ++)      				 // ע��û�е��ں�
		{
			uchar* data2 = imgCut1.ptr<uchar>(i_last);          // ptr<uchar>(i)[j]���ʵ�i�е�j�е�����
			sum_last = data2[j_last] + sum_last;
		}                                                // ����j�е�ÿһ�м���
		if (sum_last>-0.000001 && sum_last< 0.000001)            // double���ͣ�����д==0,���������������Ϳ�����������sum=0
		{
			j_last++;                                         //�����һ�ж�û������ֵ,��ô��Ȼ��û�а���LED��ROI��
		}
		else
		{
			break;										 // �������whileѭ��
		}
	}
	X_min = j_last;

	while (j_last < col_last)                                     // j�ĳ�ֵΪX_min
	{
		double sum_last = 0.0;
		for (int i_last = 0;i_last < row_last;i_last++)
		{
			uchar* data2 = imgCut1.ptr<uchar>(i_last);          // ptr<uchar>(i)[j]���ʵ�i�е�j�е�����
			sum_last = data2[j_last] + sum_last;
		}                                                // ����j�е�ÿһ�м���
		if (sum_last != 0)
		{
			j_last++;
		}
		else
		{
			break;										// �������whileѭ��
		}
	}
	X_max = j_last;                                          //����X_max��ʾ���LED��ROI���ұߵı߽�

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
		double sum_last = 0.0;    //�����������ܺ�Ϊ0
		for (int i_last = 0;i_last < row_last; i_last ++)      				 // ע��û�е��ں�
		{
			uchar* data2 = imgCut1.ptr<uchar>(i_last);          // ptr<uchar>(i)[j]���ʵ�i�е�j�е�����
			sum_last = data2[j_last] + sum_last;
			cout<<"һ�е����غ�:"<<sum_last<<endl;
		}                                                // ����j�е�ÿһ�м���
		if (sum_last < 800)            // double���ͣ�����д==0,���������������Ϳ�����������sum=0
		{
			j_last++;                                         //�����һ�ж�û������ֵ,��ô��Ȼ��û�а���LED��ROI��
		}
		else
		{
			break;										 // �������whileѭ��
		}
	}
	X_min = j_last;

	while (j_last < col_last)                                     // j�ĳ�ֵΪX_min
	{
		double sum_last = 0.0;
		for (int i_last = 0;i_last < row_last;i_last++)
		{
			uchar* data2 = imgCut1.ptr<uchar>(i_last);          // ptr<uchar>(i)[j]���ʵ�i�е�j�е�����
			sum_last = data2[j_last] + sum_last;
		}                                                // ����j�е�ÿһ�м���
		if (sum_last != 0)
		{
			j_last++;
		}
		else
		{
			break;										// �������whileѭ��
		}
	}
	X_max = j_last;                                          //����X_max��ʾ���LED��ROI���ұߵı߽�

	Mat imgCut2 = imgCut1(Rect(X_min, 0, X_max - X_min, row_last));
	thin_img_cut = imgCut2.clone();
	return thin_img_cut;
}
 
*/
