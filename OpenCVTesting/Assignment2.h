#pragma once

#include <opencv2/opencv.hpp>
#include "ClassDescriptor.h"

//Classify using the multivariate gaussian. 
cv::Mat MultivariateGaussian(cv::Mat Img, std::vector<ClassDescriptor>& Descriptors, std::vector<cv::Mat>& CovarMats, std::vector<cv::Mat>& featureImgs);
//Calculate the confusion matrix.
float  ConfusionMatrix(const cv::Mat& GroundTruth, const cv::Mat& ClassifiedImage);

void StartAssignment2();
//Compute the QFeature for a given subregion.
float ComputeQFeature(cv::Mat GLCM, size_t start_x_t, size_t end_x_t, size_t start_y_t, size_t end_y_t);
//Compute all QFeatures for the GLCM.
std::vector<float> ComputeQFeatures(cv::Mat GLCM);
//Compute the GLCM for the current image. It is not normalized.
std::vector<cv::Mat> ComputeGLCM(const cv::Mat& Img, std::vector<int> XYOffsets);
//Reduce the graylevels of the image to fit the GLCM.
void ReduceGrayLevels(cv::Mat& Img, int numGrayLevels);

std::vector<ClassDescriptor> ComputeClassDescriptors(std::vector<cv::Mat>& featureImgs, cv::Mat& mask);

std::vector<cv::Mat> ComputeQFeatureImgs(cv::Mat& Img, std::vector<int> XYOffsets);

cv::Mat NormalizeMat(cv::Mat& mat);

cv::Mat getPixelDescriptor(std::vector<cv::Mat>& featureImgs, size_t y, size_t x);

inline void PrintMat(const cv::Mat& Img)
{
	for (int y = 0; y < Img.rows; y++) {
		for (int x = 0; x < Img.cols; x++) {
			std::cout << Img.at<float>(y, x);
		}
		std::cout << "|" << std::endl;
	}
}

std::vector<std::vector<std::vector<float>>> computeSamples(cv::Mat& mask, std::vector<cv::Mat>& featureImgs);