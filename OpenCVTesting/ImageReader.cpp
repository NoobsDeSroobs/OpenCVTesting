#include "ImageReader.h"
#include <fstream>
#include <opencv2/video/tracking.hpp>

cv::Mat ReadImageFromTXT(std::string path)
{
	cv::Mat Image(5000, 5000, CV_32F);
	cv::Mat FinalImage;
	std::ifstream file(path);
	size_t x = 0;
	size_t y = 0;
	size_t xMax = 0;
	size_t yMax = 0;

	if (file.is_open())
	{
		while (file.good())
		{
			std::string line;
			getline(file, line);
			size_t pos = 0;
			std::string token;
			std::string delimiter = ",";

			while ((pos = line.find(delimiter)) != std::string::npos)
			{
				token = line.substr(0, pos);
				float value = std::stoi(token);
				Image.at<float>(y, x) = value;
				x++;
				//std::cout << token << std::endl;
				line.erase(0, pos + delimiter.length());
			}

			y++;
			if (x > xMax) xMax = x;
			if (y > yMax) yMax = y;
			x = 0;
		}

		//This will, hopefully, reduce the size to the image size.
		Image.resize(yMax-1);
		FinalImage = Image.colRange(0, xMax+1);

		file.close();
	}
	else
	{
		std::cout << "Unable to open file" << std::endl << std::endl;
	}




	float Max = 0;

	for (size_t y = 0; y < FinalImage.rows; y++) {
		for (size_t x = 0; x < FinalImage.cols; x++) {
			float Pixel = FinalImage.at<float>(y, x);

			if (Pixel > Max) Max = Pixel;
		}
	}

	/*for (size_t y = 0; y < FinalImage.rows; y++) {
		for (size_t x = 0; x < FinalImage.cols; x++) {
			float Pixel = FinalImage.at<float>(y, x);
			Pixel = (Pixel / Max)*255;
			FinalImage.at<float>(y, x) = Pixel;
		}
	}*/


	cv::Mat ConvertedImage(FinalImage.rows, FinalImage.cols, CV_8U);

	for (int i = 0; i < ConvertedImage.rows; i++)
	{
		for (int j = 0; j < ConvertedImage.cols; j++)
		{
			ConvertedImage.at<uchar>(i, j) = static_cast<uchar>(FinalImage.at<float>(i, j));
			//std::cout << ConvertedImage.at<uchar>(i, j)+48 << std::endl;
		}
	}




	cv::resize(ConvertedImage, ConvertedImage, cv::Size(100, 100));

	return ConvertedImage;
}

