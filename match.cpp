#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

#define tgtLen 128
#define tgtWidth 96

unsigned char pic1[200][200];
unsigned char pic2[200][200];

void Match(const char* pth1, const char* pth2)
{
	ifstream if1(pth1, ios::in | ios::binary);
	ifstream if2(pth2, ios::in | ios::binary);
	
	int white1=0;
	int white2=0;

	for(int i=0;i<tgtWidth;++i)
		for(int j=0;j<tgtLen;++j){
			if1.read((char*)&pic1[i][j],1);
			if2.read((char*)&pic2[i][j],1);
			if(pic1[i][j]==255) ++white1;
			if(pic2[i][j]==255) ++white2;
		}

	//if1.close();
	//if2.close();

	int maxMatch = 0;

	for(int diffW=-15;diffW<=15;++diffW)
		for(int diffL=-24;diffL<=24;++diffL){
			int count = 0;
			for(int i=0;i<96;++i)
				for(int j=0;j<128;++j)
					if(i+diffW>=0 && i+diffW<96 && j+diffL>=0 && j+diffL<128 && pic1[i+diffW][j+diffL]==255 && pic2[i][j]==255)
						++count;
			if(count>maxMatch) maxMatch = count;
		}

	double result = (maxMatch*1.0)/(white1*1.0) + (maxMatch*1.0)/(white2*1.0);
	result/=2;
	cout<<result<<endl;
}

int main(int argc, char* argv[])
{
	char* path_1 = argv[1];
	char* path_2 = argv[2];
	Match(path_1, path_2);
	return 0;
}
