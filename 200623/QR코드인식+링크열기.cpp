#include <opencv2/opencv.hpp>
#include <iostream>
#include <windows.h>
#include <shellapi.h>

using namespace cv;
using namespace std;

LPWSTR ConvertToLPWSTR(const std::string& s)
{
	// str to LPWSTR

	LPWSTR ws = new wchar_t[s.size() + 1]; // +1 for zero at the end
	
	copy(s.begin(), s.end(), ws);
	ws[s.size()] = 0; // zero at the end
	
	return ws;
}

void decode_qrcode()
{
	VideoCapture cap(0);

	if (!cap.isOpened()) 
	{
		cerr << "Camera open failed!" << endl;
		return;
	}

	QRCodeDetector detector;

	Mat frame;
	int flag = 0;

	while (1) 
	{
		cap >> frame;

		if (frame.empty()) 
		{
			cerr << "Frame load failed!" << endl;
			break;
		}

		vector<Point> points;
		String info = detector.detectAndDecode(frame, points);
	
		if (!info.empty()) 
		{
			polylines(frame, points, true, Scalar(0, 0, 255), 2);
			putText(frame, info, Point(10, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255));

			LPWSTR ws = ConvertToLPWSTR(info);

			ShellExecuteW(0, 0, ws, 0, 0, 5);

			flag = 1;
		}

		imshow("frame", frame);
		
		waitKey(1);

		if (flag == 1) break;
	}
}

int main()
{
	decode_qrcode();

	return 0;
}