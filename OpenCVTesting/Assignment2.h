#pragma once

#include <opencv2/opencv.hpp>
#include "ClassDescriptor.h"

//Classify using the multivariate gaussian. 
cv::Mat MultivariateGaussian(cv::Mat Img, std::vector<ClassDescriptor> Descriptors, std::vector<cv::Mat> CovarMats);
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

void NormalizeMat(cv::Mat& mat);