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
	//cv::calcCovarMatrix();
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

	cv::Mat TrainingMask = ReadImageFromTXT("training_mask.txt");
	cv::imshow("TestMask", TrainingMask);
	/*
	cv::Mat test = ReadImageFromTXT("mosaic1_train.txt");
	cv::line(test, cv::Point(test.cols / 2, 0), cv::Point(test.cols / 2, test.rows), cv::Scalar::all(255));
	cv::line(test, cv::Point(0, test.cols / 2), cv::Point(test.cols, test.cols / 2), cv::Scalar::all(255));

	cv::imshow("Like This?", test);*/
	cv::waitKey();

	cv::Mat Img;
	std::vector<int> offsets;
	offsets.push_back(1);
	offsets.push_back(0);
	offsets.push_back(0);
	offsets.push_back(1);

	Img = GLCM1[0];

	//Training
	//Collect the data as normal using the two directions I select and store it in a GLCM.
	//cv::Mat GLCM = ComputeGLCM(Img, offsets);
	//Calculate the features, Q1 to Qn.
	std::vector<cv::Mat> QFeatImgs = ComputeQFeatureImgs(Img, offsets);
	//Use the test mask to generate a descriptor using the N means of the N features for all M classes. We should have M descriptors now.
	std::vector<ClassDescriptor> Descriptors = ComputeClassDescriptors(QFeatImgs, TrainingMask);
	//Use the function cv::calcCovarMatrix() to calculate the covariance matrix for each class descriptor.
	std::vector<cv::Mat> CovarMats;
	CovarMats.resize(Descriptors.size(), cv::Mat());
	for (size_t i = 0; i < Descriptors.size(); i++) {
		cv::calcCovarMatrix(QFeatImgs, CovarMats[i], Descriptors[i].Descriptor, CV_COVAR_NORMAL);
	}

	//Classification
	//Use the function given in the book to calculate the probability of a pixel being class 1...M. Select the one with the highest probability. 
	//Input is the image, the class descriptors and the covariance matrices.
	//

}

float ComputeQFeature(cv::Mat GLCM, size_t start_x_t, size_t end_x_t, size_t start_y_t, size_t end_y_t)
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

			float Q = ComputeQFeature(GLCM, startX, endX, startY, endY);
			Qs.push_back(Q);
		}
	}

	return Qs;
}

cv::Mat ComputeGLCM(const cv::Mat& Img, std::vector<int> XYOffsets)
{
	int TotalMeasurements = 0;
	cv::Mat dst(Img.cols, Img.rows, CV_8UC1);
	cv::resize(Img, dst, cv::Size(dst.cols, dst.rows), 0, 0, CV_INTER_CUBIC);
	int NumGrayLevels = 16;
	ReduceGrayLevels(dst, NumGrayLevels);

	//Create and zero out matrix. I know I could use Zeroes.
	cv::Mat GLCM(NumGrayLevels, NumGrayLevels, CV_32F);
	for (int y = 0; y < GLCM.rows; y++) {
		for (int x = 0; x < GLCM.cols; x++) {
			GLCM.at<float>(y, x) = 0.0f;
		}
	}

	//These offsets represent the max offset in both directions given the values above.
	for (size_t i = 0; i < XYOffsets.size(); i += 2) {
		for (int y = 0; y < dst.rows - XYOffsets[i + 1]; y++) {
			for (int x = 0; x < dst.cols - XYOffsets[i]; x++) {
				int deltaX = XYOffsets[i];
				int deltaY = XYOffsets[i + 1];
				size_t GrayLevelBase = dst.at<uchar>(cv::Point(y, x));
				size_t GrayLevelTarget = dst.at<uchar>(cv::Point(y + deltaY, x + deltaX));

				GLCM.at<float>(GrayLevelBase, GrayLevelTarget) += 1;
				GLCM.at<float>(GrayLevelTarget, GrayLevelBase) += 1;
				TotalMeasurements = TotalMeasurements + 1;
			}
		}
	}
	/*for (int y = 0; y < GLCM.rows; y++) {
		for (int x = 0; x < GLCM.cols; x++) {
			GLCM.at<float>(y, x) = GLCM.at<float>(y, x) / TotalMeasurements;
		}
	}*/

	return GLCM;
}
void ReduceGrayLevels(cv::Mat& Img, int numGrayLevels)
{
	int numBitsToShift = 0;
	uchar b = 0 - 1;
	while (b > numGrayLevels - 1) {
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

std::vector<ClassDescriptor> ComputeClassDescriptors(std::vector<cv::Mat>& featureImgs, cv::Mat& mask)
{
	std::vector<ClassDescriptor> descriptors;

	//Find the number of classes in the mask.
	size_t numClasses = 0;
	for (size_t y_t = 0; y_t < mask.rows; y_t++) {
		for (size_t x_t = 0; x_t < mask.cols; x_t++) {
			size_t currentClassID = mask.at<float>(y_t, x_t);
			if (currentClassID > numClasses) numClasses = currentClassID;
		}
	}

	
	//Create the descriptors.
	for (size_t i = 0; i < numClasses; i++)
	{
		descriptors.push_back(ClassDescriptor(featureImgs.size()));
	}

	//Compute the class avarages.
	//For every pixel.
	for (size_t y_t = 0; y_t < mask.rows; y_t++) {
		for (size_t x_t = 0; x_t < mask.cols; x_t++) {

			int classID = mask.at<size_t>(y_t, x_t);
			std::vector<float> pixelFeatures;
			for (size_t i = 0; i < featureImgs.size(); i++) {
				//Extract all features for the pixel
				pixelFeatures.push_back(featureImgs[i].at<float>(y_t, x_t));
			}

			descriptors[classID].addPixel(pixelFeatures);
		}
	}

	for (size_t i = 0; i < descriptors.size(); i++) {
		descriptors[i].CalculateDescriptor();
	}

	return descriptors;
}

std::vector<cv::Mat> ComputeQFeatureImgs(cv::Mat& Img, std::vector<int> XYOffsets)
{
	//TODO Figure out how to determine the size of this vector.
	std::vector<cv::Mat> QFeatureIms;
	int WindowSize = 31;


	for (size_t y = 0; y < Img.rows - WindowSize; y++) {
		std::cout << "Pos: " << y << std::endl;
		for (size_t x = 0; x < Img.cols - WindowSize; x++) {
			cv::Mat Window = Img(cv::Rect(y, x, WindowSize, WindowSize));
			cv::Mat GLCM = ComputeGLCM(Window, XYOffsets);
			std::vector<float> QFeats = ComputeQFeatures(GLCM);

			for (size_t i = 0; i < QFeatureIms.size(); i++) {
				QFeatureIms[i].at<float>(x + 1 + (WindowSize / 2), y + 1 + (WindowSize / 2)) = QFeats[i];
			}
		}
	}

	

	return QFeatureIms;
}