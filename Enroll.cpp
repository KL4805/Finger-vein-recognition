//Finger vein recognition
//算法设计与分析小班Project 李文新老师
//Enroll
//1500012708 金逸伦
//二值化，旋转并加以滤波形成模板

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include "bitmap.h"
#include <cmath>
#include <opencv2\opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cstdlib>
#include <queue>
#include <vector>

using namespace std;
using namespace cv;
//int dis = 7;		//参数dis表示点对应搜索半径
//int t = 1;			//参数t表示圆周内突变值至少为t，越大对突变要求越高（灰度差越大）
//int g = 72;		//参数g表示圆周内突变情况，越小对突变要求越高（突变数越多）
//previously 6 98

int dis = 8;		//参数dis表示点对应搜索半径
int t = 1;			//参数t表示圆周内突变值至少为t，越大对突变要求越高（灰度差越大）
int g = 98;		//参数g表示圆周内突变情况，越小对突变要求越高（突变数越多）


//#define DEBUG
vector <Point> Points;
vector <Point> visited;
Vec4f midLine;     //存储中线的参数，点和方向向量
const int maxLen=550;
bool reached[maxLen][maxLen];
const int srcLen=512;
const int srcWidth=384;
const int dstLen=128;
const int dstWidth=96;
const double compRatio=0.25;
const int kernelSize=3;//Canny函数中Sobel算子的大小
const int lowThreshold = 70;
const int highThreshold = 240;
const int impArea = 12;  
//图像的上部和下部都有一段是不会有手指信息的
const int leftRed=8;
const int rightRed=4;
//图像的左侧有一部分特别亮，去掉
const int contourWidth=6; 
//我们认为边界是连续的，如果在这个范围以内没有见到一定数量边界点，那么就认为这是噪点
const double contourHeight=4;
const int midArea = 10;
//认为在图像一半高度上下一定区域内是不会有边界的
const double missPenalty = 3.5;
int avgBright=85;
double angleDev=0.5;
//prev 8 2 1.9

/*#define DEBUG
vector <Point> Points;
Vec4f midLine;     //存储中线的参数，点和方向向量
const int maxLen=550;
const int srcLen=512;
const int srcWidth=384;
const int dstLen=128;
const int dstWidth=96;
const double compRatio=0.25;
const int kernelSize=3;//Canny函数中Sobel算子的大小
const int lowThreshold = 75;
const int highThreshold = 230;
const int impArea = 12;  
//图像的上部和下部都有一段是不会有手指信息的
const int leftRed=8;
const int rightRed=4;
//图像的左侧有一部分特别亮，去掉
const int contourWidth=8; 
//我们认为边界是连续的，如果在这个范围以内没有见到一定数量边界点，那么就认为这是噪点
const double contourHeight=5;
const int midArea = 10;
//认为在图像一半高度上下一定区域内是不会有边界的
//const double missPenalty = 1.9;
//int avgBright=85;
double angleDev=0.5;*/
int claheThreshold = 68;
int highbThreshold = 90;
int adjustBright = 60; 



unsigned char color[maxLen][maxLen];
unsigned char binary[maxLen][maxLen];
int upContour[maxLen];
int downContour[maxLen];
double PI = 3.1415926535897;
double eps = 1e-6;
struct Line
{
	Point a;
	Point b;
	Line(Point _a, Point _b):a(_a),b(_b){}
};
#define Vector Point
bool isZero(double x)
{
	return -eps<x && x<eps;
}
bool fLarger(double a, double b)
{
	return a-b > eps;
}
bool fLess(double a, double b)
{
	return a-b < eps;
}
Vector operator -(Point a, Point b)
{
	return (Vector)(a.x-b.x,a.y-b.y);
}
Vector operator +(Vector a, Vector b)
{
	return (Vector)(a.x+b.x,a.y+b.y);
}
double dist(Point a, Point b)
{
	return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}
double operator *(Vector a, Vector b)
{
	return a.x*b.x+a.y*b.y;
}
Vector operator *(double k, Vector a)
{
	return Vector(a.x*k, a.y*k);
}
double length(Vector p)
{
	return sqrt(p*p);
}
Vector unit(Vector v)
{
	return 1/length(v) * v;
}
double project(Vector p, Vector n)
{
	return p*unit(n);
}
double operator ^(Vector p, Vector q)
{
	return p.x*q.y-q.x*p.y;
}

double dist(Point p, Line l)
{
	return fabs((p-l.a)^(l.b-l.a))/length(l.b-l.a);
}
Point rotate(Point b,Point a, double alpha)
{
	Vector p = b-a;
	return Point (a.x+(p.x*cos(alpha)-p.y*sin(alpha)),a.y+(p.x*sin(alpha)+p.y*cos(alpha)));
}
Line Vertical(Point p, Line l)
{
	return Line(p, p+(rotate(l.b,l.a,PI/2)-l.a));
}
Point foot(Point p, Line l)
{
	return l.a+project(p-l.a,l.b-l.a)*unit(l.b-l.a);
}


inline double Rad2Deg(double rad)
{
    return 57.295779513082323*rad;
}

Mat Compress(Mat& img)
{
    Mat comp;
    resize(img, comp, Size(0,0), compRatio, compRatio, CV_INTER_LINEAR);
    //一共有五种插值方式，NEAREST, LINEAR, AREA, CUBIC, LANCZOS4
    //NEAREST插值肯定没有LINEAR好，因为LINEAR会平滑处理，剩下的再试
#ifdef DEBUG
    namedWindow("Compression");
    imshow("Compression",comp);
	imwrite("compression.bmp", comp);
    waitKey(0);
#endif  //调试时看效果
	return comp;   
}


Mat gSmooth(Mat& img)  //高斯滤波
{
	Mat src = img;
	Mat dst;
	//高斯滤波有五个参数
	//后三个：
	//第一个Size(x,x)是高斯算子的大小，奇数
	//越大越模糊
	//然后是x方向方差和y方向方差
	GaussianBlur(src, dst, Size(3,3),1,1);
#ifdef DEBUG
	namedWindow("gSmoothed");
	imshow("gSmoothed",dst);
	imwrite("gaussian.bmp", dst);
	waitKey(0);
#endif
	return dst;
}

void Contour(Mat& img)
{
	Mat cannized;
	Canny(img, cannized, lowThreshold, highThreshold, kernelSize);
	//调用Canny边缘检测函数
	//Canny边缘检测函数有三个参数，threshold1, threshold2, kernelSize
	//kernelSize是所使用的高斯滤波矩阵的大小
	//越小，图像越清晰，可以得到更清楚的边界，但是噪点影响也更大
	//大则反之
	//两个参数分别表示通过梯度判定边界的阈值，分别是上阈值和下阈值
	//这些都要调整
#ifdef DEBUG
	namedWindow("cannized");
	imshow("cannized", cannized);
	imwrite("cannized.bmp", cannized);
	waitKey(0);
#endif

	memset(upContour, 0, sizeof(upContour));
	memset(downContour, 0, sizeof(downContour));

	bool edge[maxLen][maxLen];
	memset(edge, 0, sizeof(edge));
	for(int i=impArea;i<dstWidth-impArea;++i)
		for(int j=0;j<dstLen;++j){
		//去除不可能区域
			if(cannized.at<uchar>(i,j)==0) edge[i][j]=0;
			else edge[i][j]=1;
		}
	for(int j=dstLen-1;j>=0;--j){
		double minDev = 0;
		for(int i=0;i<=dstWidth/2-midArea;++i){
			if(edge[i][j]){
				if(j == dstLen -1)
					upContour[j]=i;
				else if(!upContour[j]){
					upContour[j]=i;
					double count = 0;
					int total = 0;
					for(int p=j+1;p<=dstLen-1 && p<=j+contourWidth;++p){
						if(p<dstLen) ++total;
						if(p<dstLen && !upContour[p]) count+=missPenalty;
						else if(p<dstLen && upContour[p]) {
							count += abs(upContour[p]-i);
							//++total;
						}
					}
					if((count*1.0)>contourHeight*total) upContour[j]=0;
					else minDev = (count*1.0)/(total*1.0);
				}
				else{
			   		double count = 0;
					int total = 0;
					for(int p=j+1;p<=dstLen-1 && p<=j+contourWidth;++p){
						if(p<dstLen) ++total;
						if(p<dstLen && !upContour[p])count += missPenalty;
						if(p<dstLen && upContour[p]) {
							count += abs(upContour[p]-i);
							//++total;
						}
					}
					if((count*1.0)/total<minDev) {
						upContour[j]=i;
						minDev=(count*1.0)/(total*1.0);
					}
				}
			}
		}
	}
	for(int j=dstLen-1;j>=0;--j){
		//cout<<"j="<<j<<endl;
		double minDev = 0;
		for(int i=dstWidth - 1;i>=dstWidth/2 + midArea;--i){
			if(edge[i][j]){
				if(j==dstLen - 1)
					downContour[j]=i;
				else if(!downContour[j]){
					downContour[j]=i;
					double count = 0;
					int total = 0;
					for(int p=j+1;p<=dstLen-1 && p<=j+contourWidth;++p){
						if(p<dstLen) ++total;
						if(p<dstLen && !downContour[p])count+=missPenalty;
						if(p<dstLen && downContour[p]) {
							count += abs(downContour[p]-i);
							//++total;
						}
					}
					//cout<<i<<' '<<(count*1.0)/(total*1.0)<<endl;
					if((count*1.0)/(total*1.0)>contourHeight) downContour[j]=0;
					else minDev = (count*1.0)/(total*1.0);
				}
				else{
					double count = 0;
					int total = 0;
					for(int p=j+1;p<=dstLen-1 && p<=j+contourWidth;++p){
						if(p<dstLen) ++total;
						if(p<dstLen && !downContour[p])count+=missPenalty;
						if(p<dstLen && downContour[p]) {
							count += abs(downContour[p]-i);
							//++total;
						}
					}
					//cout<<i<<' '<<(count*1.0)/(total*1.0)<<endl;
					if((count*1.0)/(total*1.0)<minDev) {
						downContour[j]=i;
						minDev=(count*1.0)/(total*1.0);
					}
				}
				//cout<<"contour at "<<downContour[j]<<endl;
			}
		}
	}
	Mat test = cannized;
	for(int i=0;i<dstWidth;++i)
		for(int j=0;j<dstLen;++j){
			if(i!=0 && (upContour[j]==i || downContour[j]==i))
				test.at<uchar>(i,j)=255;
			else test.at<uchar>(i,j)=0;
		}
	//考虑可能经过刚才的过滤，边界变成不连续了
	for (int i = 0; i < dstLen; i++){
		//如果轮廓开头有不连续的情况
		if (i == 0 && upContour[i] == 0){
			int j = i + 1;				//j记录后一个不空的像素点
			while (upContour[j] == 0)
				j++;

			//直接拿水平直线补全
			for (int k = 0; k < j; k++)
				upContour[k] = upContour[j];

			i = j - 1;
		}
		else if (i != 0 && upContour[i] == 0){
			int j = i + 1;				//j记录后一个不空的像素点（注意不能超过范围）
			while (j < dstLen && upContour[j] == 0)
				j++;

			if (j == dstLen){			//结尾不连续
				//用水平直线补全
				for (int k = i; k < dstLen; k++)
					upContour[k] = upContour[i - 1];
			}
			else{						//其他情况
				int diffY = j - i;		//计算y轴的像素差距
				int diffX = upContour[j] - upContour[i - 1];	//计算x差距
				double diff = (diffX*1.0)/diffY;
				for(int k=i;k<j;++k) upContour[k]=(int)(upContour[k-1]+diff+0.5);
			}
		}
	}
	//下边界同样处理
	for(int i=0;i<dstLen;++i){
		if (i == 0 && downContour[i] == 0){
			int j = i + 1;				//j记录后一个不空的像素点
			while (downContour[j] == 0)
				j++;

			//直接拿水平直线补全
			for (int k = 0; k < j; k++)
				downContour[k] = downContour[j];

			i = j - 1;
		}
		else if(i!=0 && downContour[i]==0){
			int j=i+1;
			while(j<dstLen && downContour[j]==0) ++j;
			if(j==dstLen){
				for(int k=i;k<dstLen;++k) downContour[k]=downContour[i-1];
			}
			else{
				int diffY=j-i;
				int diffX=downContour[j]-downContour[i-1];
				double diff = (diffX*1.0)/diffY;
				for(int k=i;k<j;++k) downContour[k]=(int)(downContour[k-1]+diff+0.5);
			}
		}
	}
	Mat contr = cannized;
	for(int i=0;i<dstWidth;++i)
		for(int j=0;j<dstLen;++j){
			if(i!=0 && (upContour[j]==i || downContour[j]==i))
				contr.at<uchar>(i,j)=255;
			else contr.at<uchar>(i,j)=0;
		}

#ifdef DEBUG
	namedWindow("contr");
	imshow("contr", contr);
	imwrite("contr.bmp", contr);

	waitKey(0);
#endif
}

Mat rotate(Mat& img) //旋转手指
{
	Mat rot = img;
	for(int i=0;i<dstLen;i+=2)
		Points.push_back(Point(i,(upContour[i]+downContour[i])/2.0));
	fitLine(Mat(Points), midLine, CV_DIST_L2, 0, 0.01, 0.01);
	//fitLine函数的参数分别是：原始点集，Vec4f的对象
	//CV_DIST_L2表示最小二乘法
	//456参数opencv推荐使用0, 0.01, 0.01
	//得到的midLine[0][1]是一个归一化的方向向量
	//[2][3]是直线上一个点

	Mat rot_mat(2,3,CV_8UC1);
	Point center = Point(midLine[3], midLine[2]);
	double angle = Rad2Deg((double)(asin(midLine[1])+acos(midLine[0])+atan(midLine[1]/midLine[0]))/3.0);
	if(angle<0) angleDev = -1*angleDev;
	rot_mat = getRotationMatrix2D(center, angle+angleDev, 1);
	//三个参数，中心，旋转角，缩放比
	warpAffine(img, rot, rot_mat, img.size());
	//做一个仿射变化，四个参数分别为源，目的，变换矩阵，图像大小
	return rot;
}

Mat Gamma(double exp, Mat img)
{
	Mat temp=img;
	int lut[268]={0};
	double ceiling = pow(255, exp);
	double normal = 255.0/ceiling;
	for(int i=0;i<=255;++i) lut[i]=(int)(0.5+normal*pow(i,exp));
	for(int i=0;i<dstWidth;++i)
		for(int j=0;j<dstLen;++j) temp.at<uchar>(i,j)=lut[temp.at<uchar>(i,j)];
#ifdef DEBUG
	namedWindow("gamma");
	imshow("gamma", temp);
	imwrite("gamma.bmp",temp); 
	waitKey(0);
#endif
	return temp;
}

void Enroll(char* source, char* dest)
{
	Mat original;
	original = imread(source, 0);
	int totalbright = 0;
	Mat img = Compress(original);
	gSmooth(img);
	for(int i=0;i<dstWidth;++i)
		for(int j=0;j<dstLen;++j)
			totalbright += img.at<uchar>(i,j);
	double avgbright = (totalbright*1.0)/(128.0*96.0);
	
	//if(avgbright<claheThreshold){
	/*for(int i=0;i<dstWidth;++i)
		for(int j=0;j<dstLen;++j){
			int temp =img.at<uchar>(i,j)*(avgbright/avgbright);
			img.at<uchar>(i,j)=temp>255?255:temp;
		}*/

	//Clahe增强
		Mat clahed=img;
		Ptr<CLAHE> clahe = createCLAHE();
		clahe->setClipLimit(2);
		clahe->apply(clahed, img);
		//img = Gamma(0.5,img);
	//}
	
	/*if(avgbright >= highbThreshold){
		for(int i=0;i<dstWidth;++i)
			for(int j=0;j<dstLen;++j){
				int temp =img.at<uchar>(i,j)*((adjustBright-10)/avgbright);
				img.at<uchar>(i,j)=temp>255?255:temp;
			}
		Mat clahed = img;
		Ptr<CLAHE> clahe = createCLAHE();
		clahe->setClipLimit(2);
		clahe->apply(clahed, img);
		img = Gamma(2.5,img);
	}*/


#ifdef DEBUG
	namedWindow("clahe");
	imshow("clahe",img);
	waitKey(0);
	imwrite("clahed.bmp",img);
#endif
	Contour(img);
	Mat rotated = rotate(img);
	for(int i=0;i<dstWidth;++i)
		for(int j=0;j<dstLen;++j)
			color[i][j]=img.at<uchar>(i,j);
	for(int i=0;i<dstWidth;++i)
		for(int j=0;j<dstLen;++j){
			if(upContour[j]>=i || downContour[j]<=i){
				binary[i][j]=0;
				continue;
			}
			int count = 0;
			for(int di=0;di<=dis;++di)
				for(int dj=0;dj*dj+di*di<=dis*dis;++dj){
					if(color[i+di][j+dj]-color[i][j]<=t) ++count;
					if(di>0 && i-di>=0 && color[i-di][j+dj]-color[i][j]<=t) ++count;
					if(dj>0 && j-dj>=0 && color[i+di][j-dj]-color[i][j]<=t) ++count;
					if(di>0 && dj>0 && i-di>=0 && j-dj>=0 && color[i-di][j-dj]-color[i][j]<=t) ++count;
				}
			if(count<=g) binary[i][j]=255;
			else binary[i][j]=0;
		}
	for(int k=0;k<leftRed;++k)
		for(int i=0;i<dstWidth;++i){
			binary[i][k]=0;
		}
	for(int k=0;k<rightRed;++k)
		for(int i=0;i<dstWidth;++i){
			binary[i][k]=0;
		}

	Mat binarized=img;
	for(int i=0;i<dstWidth;++i)
		for(int j=0;j<dstLen;++j)
			binarized.at<uchar>(i,j)=binary[i][j];
	
#ifdef DEBUG
	namedWindow("binary");
	imshow("binary", binarized);
	imwrite("binary.bmp", binarized);
	waitKey(0);
#endif
	ofstream outfile(dest, ios::out|ios::binary);
	for(int i=0;i<dstWidth;++i){
		for(int j=0;j<dstLen;++j){
			unsigned char temp = binarized.at<uchar>(i,j);
			//cout<<(int)temp<<' ';
			outfile.write((char*)&temp, sizeof(unsigned char));
		}
		//cout<<endl;
	}
	outfile.close();
}



int main(int argc, char *argv[])
{
	//char src_path[] = "0.bmp";
	//char dst_path[] = "2.txt";
	//for(int i=0;i<5;++i){				
	char *src_path = argv[1];
	char *dst_path = argv[2];
		//src_path[0]++;
		Enroll(src_path, dst_path);
	//}
	return 0;
}
