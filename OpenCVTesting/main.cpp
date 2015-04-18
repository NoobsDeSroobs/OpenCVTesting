#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>

struct Vector2D
{
	float x;
	float y;
	Vector2D(float inx, float iny) :x(inx), y(iny)
	{

	}

	Vector2D operator-(Vector2D b)
	{
		return Vector2D(x - b.x, y - b.y);
	};

	Vector2D operator+(Vector2D b)
	{
		return Vector2D(x + b.x, y + b.y);
	};

	float Length()
	{
		return std::sqrtf(std::pow(x, 2) + std::pow(y, 2));
	}
};

//These are the functions that I am thinking about making, Mitch. Any ideas? Wanna help?
float CalculateSin(int X, float YOffset, float XOffset, float MagnitudeModifier, float PeriodModifier)
{
	float alpha = MagnitudeModifier*sin(PeriodModifier*(X + XOffset)) + YOffset;
	return alpha;
}

float CalculateCos(int X, float YOffset, float XOffset, float MagnitudeModifier, float PeriodModifier)
{
	float alpha = MagnitudeModifier*cos(PeriodModifier*(X + XOffset)) + YOffset;
	return alpha;
}

void DrawSinCurve(cv::Mat& SinCurve, float MidLine, float YOffset = 0, float XOffset = 0, float MagnitudeModifier = 310.0f, float PeriodModifier = 1 / (M_PI * 73))
{
	for (float X = 0; X < SinCurve.cols; ++X) {
		float alpha = CalculateSin(X, YOffset, XOffset, MagnitudeModifier, PeriodModifier);//Range -PI to PI
		if (-alpha + MidLine > SinCurve.rows || -alpha + MidLine < 1) continue;
		SinCurve.at<uchar>(-alpha + MidLine, X) = 0.0f;
	}
}

void DrawCosCurve(cv::Mat& CosCurve, float MidLine, float YOffset = 0, float XOffset = 0, float MagnitudeModifier = 310.0f, float PeriodModifier = 1 / (M_PI * 73))
{
	for (float X = 0; X < CosCurve.cols; ++X) {
		float alpha = CalculateCos(X, YOffset, XOffset, MagnitudeModifier, PeriodModifier);//Range -PI to PI
		if (-alpha + MidLine > CosCurve.rows || -alpha + MidLine < 1) continue;
		CosCurve.at<uchar>(-alpha + MidLine, X) = 0.0f;
	}
}

void DrawFourierCurves(cv::Mat& Curve, float MidLine, float YOffset = 0, float XOffset = 0, float MagnitudeModifier = 310.0f, float PeriodModifier = 1 / (M_PI * 73))
{
	for (float X = 0; X < Curve.cols; ++X) {
		float Total = 0.0f;
		float alpha = CalculateSin(X, YOffset, XOffset, MagnitudeModifier, PeriodModifier * 9);//Range -PI to PI
		if (-alpha + MidLine <= Curve.rows || -alpha + MidLine > 0)
			Curve.at<uchar>(-alpha + MidLine, X) = 230.0f;
		Total += alpha;
		alpha = CalculateSin(X, YOffset, XOffset, MagnitudeModifier * 2, PeriodModifier * 3);//Range -PI to PI
		if (-alpha + MidLine <= Curve.rows || -alpha + MidLine > 0)
			Curve.at<uchar>(-alpha + MidLine, X) = 230.0f;
		Total += alpha;
		alpha = CalculateSin(X, YOffset, XOffset, MagnitudeModifier / 2, PeriodModifier * 8);//Range -PI to PI
		if (-alpha + MidLine <= Curve.rows || -alpha + MidLine > 0)
			Curve.at<uchar>(-alpha + MidLine, X) = 230.0f;
		Total += alpha;
		if (-Total + MidLine <= Curve.rows || -Total + MidLine > 0)
			Curve.at<uchar>(-Total + MidLine, X) = 0.0f;
	}
}

void DrawWeierstrassFunction(cv::Mat& WeierstrassCurve, float MidLine, float YOffset = 0, float XOffset = 0, float MagnitudeModifier = 10.0f, float PeriodModifier = 1 / (M_PI))
{
	float a = 0.9f;
	float b = 7.0f;

	for (float X = 0; X < WeierstrassCurve.cols; ++X) {
		float alpha = 0.0f;
		for (size_t n = 0; n < 100; n++) {
			alpha += MagnitudeModifier*pow(a, n)*cos(PeriodModifier*pow(b, n)*M_PI*(X / 72));
		}
		if (-alpha + MidLine > WeierstrassCurve.rows || -alpha + MidLine < 1) continue;
		WeierstrassCurve.at<uchar>(-alpha + MidLine, X) = 0.0f;
	}
}

void DrawMaze(cv::Mat& MazeImage, bool* MazeMap)
{

}

void DrawWithMask()
{
	cv::Mat LowerImage = cv::imread("E:/Dropbox/IMERSO/TestImLower.jpg");
	cv::Mat Mask = cv::imread("E:/Dropbox/IMERSO/MaskGradient.jpg");
	cv::cvtColor(Mask, Mask, cv::COLOR_BGR2GRAY);
	cv::Mat UpperImage = cv::imread("D:/Music, vids and documents/Pics/ANIME/1273609 - Captain-T Elsa Frozen.jpg");

	cv::Mat MaskedImage(LowerImage);
	for (size_t y = 0; y < LowerImage.rows; y++) {
		for (size_t x = 0; x < LowerImage.cols; x++) {
			float Alpha = Mask.at<uchar>(y, x) / 255.0f;
			float Beta = (1.0f - Alpha);
			MaskedImage.at<cv::Vec3b>(y, x) = LowerImage.at<cv::Vec3b>(y, x) * Alpha + Beta*UpperImage.at<cv::Vec3b>(y, x);
		}
	}

	cv::imshow("Random window", MaskedImage);

}

void DrawPoint(cv::Mat& Img, int XCenter, int YCenter, float radius)
{
	int StartX = XCenter - (int)radius, StartY = YCenter - (int)radius;
	int EndX = XCenter + (int)radius, EndY = YCenter + (int)radius;

	for (size_t y = StartY; y < EndY; y++) {
		for (size_t x = StartX; x < EndX; x++) {
			if (sqrt(pow(abs((float)XCenter - x), 2) + pow(abs((float)YCenter - y), 2)) < radius)
				if (y > 0 && x > 0 && y < Img.rows && x < Img.cols)
					Img.at<uchar>(y, x) = 0.0f;
		}
	}
}

void DrawLine(cv::Mat& Img, Vector2D Start, Vector2D End)
{
	float Step = 1 / (End - Start).Length();

	for (float i = 0; i < 1; i += Step) {
		float alphax = i * Start.x + (1 - i) * End.x;
		float alphay = i * Start.y + (1 - i) * End.y;
		if (alphay > 0 && alphax > 0 && alphay < Img.rows && alphax < Img.cols)
			Img.at<uchar>(alphay, alphax) = 0.0f;
	}
}

void MitchellVerticalGradiant(cv::Mat Img)
{
	float negroBabies = 0;
	float increaseNegros = 255.0f / 200.0f;
	int lowerLimit = 0;
	int upperLimit = 600;
	for (int i = 0; i < 600; i++) {
		if (i >= lowerLimit && i <= upperLimit) {
			negroBabies = negroBabies + increaseNegros;
		}
		for (int j = 0; j < 800; j++) {
			Img.at<uchar>(i, j) = negroBabies;
		}
	}
}

void MitchellHorizontalGradiant(cv::Mat Img)
{
	float negroBabies = 0;
	float increaseNegros = 255.0f / 800.0f;
	for (int i = 0; i < 800; i++) {
		negroBabies = negroBabies + increaseNegros;
		for (int j = 0; j < 600; j++) {
			Img.at<uchar>(j, i) = negroBabies;
		}
	}
}

void HorizontalOrVertical(cv::Mat Img, bool Horizontal)
{
	if (Horizontal) {
		MitchellHorizontalGradiant(Img);
	} else {
		MitchellVerticalGradiant(Img);
    }
}

void MitchellsMain()
{
	cv::Mat Img(600, 800, CV_8U);
	HorizontalOrVertical(Img, false);
	cv::imshow("MyWindow", Img);
}

int main()
{
	MitchellsMain();
	cv::Mat Img(M_PI * 200, 360 * 4, CV_8U);
	for (int y = 0; y < Img.rows; y += 1) {
		for (int x = 0; x < Img.cols; x++) {
			Img.at<uchar>(y, x) = 255.0f;//White backgrund
		}
	}

	float MidLine = (float)Img.rows / 2.0f;
	for (float X = 0; X < Img.cols; ++X) {
		Img.at<uchar>(MidLine, X) = 0.0f;//Black line
	}

	std::vector<Vector2D> points;

	float Coords[4];
	Coords[0] = 0.0f; // StartX
	Coords[1] = MidLine+50.0f; // StartY
	Coords[2] = Img.cols; // EndX
	Coords[3] = MidLine+50.0f; // EndY
	int k = 0;
	for (float i = 0; i < 1; i += 0.01f) {
		float DeltaX = Coords[2] - Coords[0];
		float DeltaY = Coords[3] - Coords[1];

		float alphax = i * Coords[0] + (1 - i) * Coords[2];
		float alphay = i * Coords[1] + (1 - i) * Coords[3];
		points.push_back(Vector2D(alphax + 20, alphay + (std::rand() % 80) - 40));
		DrawPoint(Img, points[k].x, points[k].y, 5.0f);
		k++;
	}

	DrawLine(Img, Vector2D(Coords[0], Coords[1]), Vector2D(Coords[2], Coords[3]));
	//cv::erode(Img, Img, cv::Mat());
	//cv::imshow("Window", Img);
	cv::waitKey(NULL);

	return 0;
}