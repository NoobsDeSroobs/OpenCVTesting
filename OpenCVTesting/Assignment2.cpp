#include "Assignment2.h"

float ConfusionMatrix(const cv::Mat& GroundTruth, const cv::Mat& ClassifiedImage)
{
	size_t numClasses = 4;
	cv::Mat ConfMat(numClasses, numClasses, CV_32F);

	for (size_t y = 0; y < GroundTruth.rows; y++)
	{
		for (size_t x = 0; x < GroundTruth.cols; x++)
		{
			ConfMat.at<float>(GroundTruth.at<float>(x, y), ClassifiedImage.at<float>(x, y))++;
		}
	}

	float Accuracy = -1;

	float Total = 0;
	float NumCorrect = 0;

	for (size_t y = 0; y < ConfMat.rows; y++) {
		for (size_t x = 0; x < ConfMat.cols; x++) {
			if (x == y)
			{
				NumCorrect += ConfMat.at<float>(x, y);
			}

			Total += ConfMat.at<float>(x, y);
		}
	}

	Accuracy = NumCorrect / Total;
	return Accuracy;
}


void MultivariateGaussian()
{
	
}
