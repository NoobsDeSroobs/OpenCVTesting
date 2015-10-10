#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>

void ReduceGrayLevels(cv::Mat& Img, int numGrayLevels)
{
	int numBitsToShift = 0;
	uchar b = 0-1;
	while (b > numGrayLevels-1) {
		b = b >> 1;
		numBitsToShift++;
	}

	int height = Img.rows;
	int width = Img.cols;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			uchar oldVal = Img.at<uchar>(y, x);
			uchar newVal = oldVal >> numBitsToShift;
			Img.at<uchar>(y, x) = newVal;
		}
	}
}


void ExtractSubImages(const cv::Mat& SourceImage, size_t ImWidth, size_t ImHeight, std::vector<cv::Mat>& OutVector)
{
	OutVector.push_back(SourceImage(cv::Rect(0, 0, ImWidth / 2, ImHeight / 2)));
	OutVector.push_back(SourceImage(cv::Rect(ImWidth / 2, 0, ImWidth / 2, ImHeight / 2)));
	OutVector.push_back(SourceImage(cv::Rect(0, ImHeight / 2, ImWidth / 2, ImHeight / 2)));
	OutVector.push_back(SourceImage(cv::Rect(ImWidth / 2, ImHeight / 2, ImWidth / 2, ImHeight / 2)));
}

cv::Mat ComputeGLCM(const cv::Mat& Img, std::vector<int> XYOffsets)
{
	int TotalMeasurements = 0;
	cv::Mat dst(Img.cols, Img.rows, CV_8UC1);
	cv::resize(Img, dst, cv::Size(dst.cols, dst.rows), 0, 0, CV_INTER_CUBIC);
	int NumGrayLevels = 32;
	ReduceGrayLevels(dst, NumGrayLevels);

	cv::Mat GLCM(NumGrayLevels, NumGrayLevels, CV_32F);
	for (int y = 0; y < GLCM.rows; y++) {
		for (int x = 0; x < GLCM.cols; x++) {
				GLCM.at<float>(x, y) = 0.0f;
		}
	}

	//These magic numbers represent the max offset in both directions given the values above.
	for (int y = 0; y < dst.rows - 3; y++) {
		for (int x = 0; x < dst.cols - 3; x++) {
			for (size_t i = 0; i < XYOffsets.size(); i+=2)
			{
				int deltaX = XYOffsets[i];
				int deltaY = XYOffsets[i + 1];
				size_t GrayLevelBase = dst.at<uchar>(cv::Point(x, y));
				size_t GrayLevelTarget = dst.at<uchar>(cv::Point(x + deltaX, y + deltaY));

				GLCM.at<float>(GrayLevelBase, GrayLevelTarget) += 1;
				GLCM.at<float>(GrayLevelTarget, GrayLevelBase) += 1;
				TotalMeasurements = TotalMeasurements + 1;
			}
		}
	}

	for (int y = 0; y < GLCM.rows; y++) {
		for (int x = 0; x < GLCM.cols; x++) {
			GLCM.at<float>(x, y) = GLCM.at<float>(x, y) / TotalMeasurements;
		}
	}

	return GLCM;
}

void PrintMat(const cv::Mat& Img){
	for (int y = 0; y < Img.rows; y++) {
		std::cout << "|";
		for (int x = 0; x < Img.cols; x++) {
			std::cout << "|" << Img.at<float>(y, x);
		}
		std::cout << "|" << std::endl;
		for (int x = 0; x < Img.cols; x++) {
			std::cout << "--";
		}
		std::cout << "|" << std::endl;
	}
}

float CalculateHomogeneity(cv::Mat& GLCM)
{
	float total = 0;
	for (size_t y = 0; y < GLCM.rows; y++)
	{
		for (size_t x = 0; x < GLCM.cols; x++)
		{
			float dividend = GLCM.at<float>(x, y);
			float divisor = (1 + ((x - y) * (x - y)));

			total += dividend / divisor;
		}
	}
	return total;
}

float CalculateClusterShade(cv::Mat& GLCM){
	float total = 0;

	float ux = 0;
	float uy = 0;

	for (size_t y2 = 0; y2 < GLCM.rows; y2++)
	{
		for (size_t x2 = 0; x2 < GLCM.cols; x2++)
		{
			ux += x2 * GLCM.at<float>(x2, y2);
			uy += y2 * GLCM.at<float>(x2, y2);
		}
	}

	for (size_t y = 0; y < GLCM.rows; y++)
	{
		for (size_t x = 0; x < GLCM.cols; x++)
		{
			float val = x + y - ux - uy;
			total += val * val * val * GLCM.at<float>(x, y);
 		}
	}

	return total;
}

float CalculateInertia(cv::Mat& GLCM){
	float total = 0;
	
	for (size_t y = 0; y < GLCM.rows; y++)
	{
		for (size_t x = 0; x < GLCM.cols; x++)
		{
			total += ((x - y)*(x - y)) * GLCM.at<float>(x, y);
		}
	}
	return total;
}

std::vector<cv::Mat> ComputeFeatures(cv::Mat& Img, std::vector<int>& XYOffsets, std::vector<float>& Avarages){
	std::vector<cv::Mat> vecFeat;
	vecFeat.reserve(3);

	cv::Mat HomoFeatures(Img.cols, Img.rows, CV_32F);
	cv::Mat InertFeatures(Img.cols, Img.rows, CV_32F);
	cv::Mat ShadeFeatures(Img.cols, Img.rows, CV_32F);
	int WindowSize = 35;


	float RAvarage = 0, GAvarage = 0, BAvarage = 0;

	for (size_t y = 0; y < Img.rows - WindowSize; y++)
	{
		std::cout << "Pos: " << y << std::endl;
		for (size_t x = 0; x < Img.cols - WindowSize; x++)
		{
			cv::Mat Window = Img(cv::Rect(x, y, WindowSize, WindowSize));
			cv::Mat GLCM = ComputeGLCM(Window, XYOffsets);
			float Homo = CalculateHomogeneity(GLCM);
			float Inert = CalculateInertia(GLCM);
			float Shade = abs(CalculateClusterShade(GLCM));

			//cv::Vec3b Pixel = cv::Vec3b(Homo, Inert, ClustShad);
			HomoFeatures.at<float>(x + 1 + (WindowSize / 2), y + 1 + (WindowSize / 2)) = Homo;
			InertFeatures.at<float>(x + 1 + (WindowSize / 2), y + 1 + (WindowSize / 2)) = Inert;
			ShadeFeatures.at<float>(x + 1 + (WindowSize / 2), y + 1 + (WindowSize / 2)) = Shade;
		}
	}

	float RMax = 0, GMax = 0, BMax = 0;

	for (size_t y = (WindowSize / 2)+1; y < Img.rows - (WindowSize / 2)-1; y++)
	{
		std::cout << "Pos: " << y << std::endl;
		for (size_t x = (WindowSize / 2)+1; x < Img.cols - (WindowSize / 2)-1; x++)
		{
			float HomoPixel = HomoFeatures.at<float>(x, y);
			float InertPixel = InertFeatures.at<float>(x, y);
			float ShadePixel = ShadeFeatures.at<float>(x, y);

			if (HomoPixel > RMax) RMax = HomoPixel;
			if (InertPixel > GMax) GMax = InertPixel;
			if (ShadePixel > BMax) BMax = ShadePixel;
		}
	}

	for (size_t y = (WindowSize / 2) + 1; y < Img.rows - (WindowSize / 2) - 1; y++)
	{
		std::cout << "Pos: " << y << std::endl;
		for (size_t x = (WindowSize / 2) + 1; x < Img.cols - (WindowSize / 2) - 1; x++)
		{
			float HomoPixel = HomoFeatures.at<float>(x, y);
			HomoPixel = (255.0f / RMax) * HomoPixel;
			HomoFeatures.at<float>(x, y) = HomoPixel;
	
			float InertPixel = InertFeatures.at<float>(x, y);
			InertPixel = (255.0f / GMax) * InertPixel;
			InertFeatures.at<float>(x, y) = InertPixel;

			float ShadePixel = ShadeFeatures.at<float>(x, y);
			ShadePixel = (255.0f / BMax) * ShadePixel;
			ShadeFeatures.at<float>(x, y) = ShadePixel;


			RAvarage += HomoPixel; GAvarage += InertPixel; BAvarage += ShadePixel;

		}
	}
	cv::Mat test(Img.cols, Img.rows, CV_8UC1);
	cv::Mat test2(Img.cols, Img.rows, CV_8UC1);
	cv::Mat test3(Img.cols, Img.rows, CV_8UC1);
	for (size_t y = (WindowSize / 2) + 1; y < Img.rows - (WindowSize / 2) - 1; y++)
	{
		for (size_t x = (WindowSize / 2) + 1; x < Img.cols - (WindowSize / 2) - 1; x++)
		{
			test.at<uchar>(x, y) = (uchar)HomoFeatures.at<float>(x, y);
			test2.at<uchar>(x, y) = (uchar)InertFeatures.at<float>(x, y);
			test3.at<uchar>(x, y) = (uchar)ShadeFeatures.at<float>(x, y);

		}
	}

	vecFeat.push_back(test);
	vecFeat.push_back(test2);
	vecFeat.push_back(test3);

	RAvarage = RAvarage / ((Img.rows - WindowSize)*(Img.cols - WindowSize));
	GAvarage = GAvarage / ((Img.rows - WindowSize)*(Img.cols - WindowSize));
	BAvarage = BAvarage / ((Img.rows - WindowSize)*(Img.cols - WindowSize));
	Avarages.push_back(RAvarage);
	Avarages.push_back(GAvarage);
	Avarages.push_back(BAvarage);
	std::cout << "Homo avarage: " << RAvarage << "Inert avarage: " << GAvarage << "Shade avarage: " << BAvarage << std::endl;

	return vecFeat;
}

std::vector<cv::Mat> SegmentThresholdImage(std::vector<cv::Mat> FeatImgs, std::vector<float>& Thresholds){
	std::vector<cv::Mat> Masks;
	for (size_t i = 0; i < FeatImgs.size(); i++)
	{
		cv::Mat FeatImg = FeatImgs[i];
		cv::Mat Mask(FeatImg.cols, FeatImg.rows, CV_8UC1);
		uchar Threshold = Thresholds[i];
		for (size_t y = 0; y < FeatImg.rows; y++)
		{
			for (size_t x = 0; x < FeatImg.cols; x++)
			{
				if (FeatImg.at<uchar>(x, y) > Threshold){
					Mask.at<uchar>(x, y) = 255;
				}else{
					Mask.at<uchar>(x, y) = 0;
				}
			}
		}
		
		Masks.push_back(Mask);

	}

	return Masks;

}

int main()
{

	cv::Mat BaseImage2 = cv::imread("mosaic1.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (!BaseImage2.data)// Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	cv::Mat BaseImage1 = cv::imread("mosaic2.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (!BaseImage1.data)// Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	std::vector<cv::Mat> SubImages1;
	std::vector<cv::Mat> SubImages2;

	ExtractSubImages(BaseImage1, BaseImage1.cols, BaseImage1.rows, SubImages1);
	ExtractSubImages(BaseImage2, BaseImage2.cols, BaseImage2.rows, SubImages2);

	cv::Mat ScaledImg;
	cv::resize(BaseImage1, ScaledImg, cv::Size(BaseImage1.cols / 1, BaseImage1.rows / 1));

	std::vector<int> XYOffsets;
	XYOffsets.push_back(0);
	XYOffsets.push_back(3);
	XYOffsets.push_back(3);
	XYOffsets.push_back(0);

	//std::vector<float> Avarages;
	//std::vector<cv::Mat> FeatImgs = ComputeFeatures(ScaledImg, XYOffsets, Avarages);
	//std::vector<cv::Mat> FeatMasks = SegmentThresholdImage(FeatImgs, Avarages);
	//cv::Mat finalImage;
	//cv::merge(FeatMasks, finalImage);
	//cv::Mat ColourImg;
	//cv::cvtColor(ScaledImg, ColourImg, CV_GRAY2BGR);
	//ColourImg = ColourImg * 0.6 + finalImage * 0.4;

	//cv::imshow("Homo.", FeatMasks[0]);
	//cv::imshow("Inert.", FeatMasks[1]);
	//cv::imshow("Shade.", FeatMasks[2]);
	//cv::imshow("Overlay.", ColourImg);
	//cv::imshow("Masks as RGB.", finalImage);


	//cv::imwrite("HomoI1S35.jpg", FeatMasks[0]);
	//cv::imwrite("InertI1S35.jpg", FeatMasks[1]);
	//cv::imwrite("ShadeI1S35.jpg", FeatMasks[2]);
	//cv::imwrite("OverlayI1S35.jpg", ColourImg);
	//cv::imwrite("Masks as RGBI1S35.jpg", finalImage);

	for (size_t i = 0; i < 4; i++) {

		cv::Mat GLCM = ComputeGLCM(SubImages1[i], XYOffsets);
		//PrintMat(GLCM);
		std::cout << "Img " << i+1 << " has these values:" << std::endl;
		std::cout << "    " << "Homogeneity: " << CalculateHomogeneity(GLCM) << std::endl;
		std::cout << "    " << "Cluster Shade: " << CalculateClusterShade(GLCM) << std::endl;
		std::cout << "    " << "Inertia: " << CalculateInertia(GLCM) << std::endl;
	}

	cv::imshow("Img1: ", SubImages1[0]);
	cv::imshow("Img2: ", SubImages1[1]);
	cv::imshow("Img3: ", SubImages1[2]);
	cv::imshow("Img4: ", SubImages1[3]);

	cv::waitKey(0);
	return 0;
}