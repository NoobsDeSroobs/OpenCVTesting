#include "Assignment2.h"
#include "ImageReader.h"

float ConfusionMatrix(const cv::Mat& GroundTruth, const cv::Mat& ClassifiedImage)
{
	size_t numClasses = 4;
	cv::Mat ConfMat(numClasses, numClasses, CV_32F);

	for (size_t y = 0; y < GroundTruth.rows; y++)
	{
		for (size_t x = 0; x < GroundTruth.cols; x++)
		{
			ConfMat.at<float>(GroundTruth.at<float>(y, x), ClassifiedImage.at<float>(y, x))++;
		}
	}

	float Accuracy = -1;

	float Total = 0;
	float NumCorrect = 0;

	for (size_t y = 0; y < ConfMat.rows; y++) {
		for (size_t x = 0; x < ConfMat.cols; x++) {
			if (x == y)
			{
				NumCorrect += ConfMat.at<float>(y, x);
			}

			Total += ConfMat.at<float>(y, x);
		}
	}

	Accuracy = NumCorrect / Total;
	return Accuracy;
}


void MultivariateGaussian()
{
	
}


void StartAssignment2()
{
	std::vector<cv::Mat> GLCM1;
	std::vector<cv::Mat> GLCM2;
	GLCM1.push_back(ReadImageFromTXT("texture1dx0dy1.txt"));
	GLCM1.push_back(ReadImageFromTXT("texture2dx0dy1.txt"));
	GLCM1.push_back(ReadImageFromTXT("texture3dx0dy1.txt"));
	GLCM1.push_back(ReadImageFromTXT("texture4dx0dy1.txt"));

	GLCM2.push_back(ReadImageFromTXT("texture1dx1dy0.txt"));
	GLCM2.push_back(ReadImageFromTXT("texture2dx1dy0.txt"));
	GLCM2.push_back(ReadImageFromTXT("texture3dx1dy0.txt"));
	GLCM2.push_back(ReadImageFromTXT("texture4dx1dy0.txt"));

	/*
	cv::Mat test = ReadImageFromTXT("mosaic1_train.txt");
	cv::line(test, cv::Point(test.cols / 2, 0), cv::Point(test.cols / 2, test.rows), cv::Scalar::all(255));
	cv::line(test, cv::Point(0, test.cols / 2), cv::Point(test.cols, test.cols / 2), cv::Scalar::all(255));

	cv::imshow("Like This?", test);*/
	cv::waitKey();

	//Collect the data as normal using the two directions I select and store it in a GLCM.
	//Calculate the features, Q1 to Qn.
	//Use the test mask to generate a descriptor using the N means of the N features for all M classes. We should have M descriptors now.
	//Use the function cv::calcCovarMatrix() to calculate the covariance matrix for each class descriptor.
	//Us ethe function given in the book to calculate the probability of a pixel being class 1...M. Select the one with the highest probability.
	//

}

float ComputeFeature(cv::Mat GLCM, size_t start_x_t, size_t end_x_t, size_t start_y_t, size_t end_y_t)
{
	float Q = 0;
	float SubSum = 0;
	float TotalSum = 0;

	//This could have been calculated once, but fuck it. It is not that time demanding.
	for (size_t y_t = 0; y_t < GLCM.rows; y_t++) {
		for (size_t x_t = 0; x_t < GLCM.cols; x_t++) {
			TotalSum += GLCM.at<float>(y_t, x_t);
		}
	}


	for (size_t y_t = start_y_t; y_t < end_y_t; y_t++) {
		for (size_t x_t = start_x_t; x_t < end_x_t; x_t++) {
			SubSum += GLCM.at<float>(y_t, x_t);
		}
	}

	Q = SubSum / TotalSum;
	return Q;
}

//Compute subGLCM features.
std::vector<float> ComputeQFeatures(cv::Mat GLCM)
{
	std::vector<float> Qs;
	size_t width_t = GLCM.cols;
	size_t height_t = GLCM.rows;
	size_t wStep = width_t / 2;
	size_t hStep = height_t / 2;

	for (size_t i = 0; i < 2; i++) {
		for (size_t j = 0; j < 2; j++) {
			size_t startX = wStep*j, endX = wStep + wStep*j, startY = hStep*i, endY =hStep + hStep*i;

			float Q = ComputeFeature(GLCM, startX, endX, startY, endY);
			Qs.push_back(Q);
		}
	}


	return Qs;
}