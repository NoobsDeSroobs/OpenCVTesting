#include "Assignment2.h"
#include "ImageReader.h"
#define _USE_MATH_DEFINES
#include <math.h>

float ConfusionMatrix(const cv::Mat& GroundTruth, const cv::Mat& ClassifiedImage)
{
	size_t numClasses = 5;
	cv::Mat ConfMat = cv::Mat::zeros(numClasses, numClasses, CV_32F);

	for (size_t y = 0; y < GroundTruth.rows; y++)
	{
		for (size_t x = 0; x < GroundTruth.cols; x++)
		{
			ConfMat.at<float>(GroundTruth.at<uchar>(y, x), ClassifiedImage.at<uchar>(y, x))++;
		}
	}

	// ReSharper disable once CppInitializedValueIsAlwaysRewritten
	float Accuracy = -1;

	float Total = 0;
	float NumCorrect = 0;

	for (size_t y = 1; y < ConfMat.rows; y++) {
		for (size_t x = 1; x < ConfMat.cols; x++) {
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


cv::Mat MultivariateGaussian(cv::Mat Img, std::vector<ClassDescriptor>& Descriptors, std::vector<std::vector<std::vector<float>>>& samples, std::vector<cv::Mat>& featureImgs)
{

	cv::Mat ClassificationMask(Img.rows, Img.cols, CV_8U);
	for (size_t y_t = 0; y_t < Img.rows; y_t++) {
		for (size_t x_t = 0; x_t < Img.cols; x_t++) {

			float d = Descriptors.size();
			std::vector<float> ProbabilityForClass(d, 0);
			//Calculate the probability for all the classes. 
			for (size_t i = 1; i < d; i++) {
				cv::Mat x = getPixelDescriptor(featureImgs, y_t, x_t);
				cv::Mat myu = Descriptors[i].Descriptor;
				cv::Mat sigma;
				cv::calcCovarMatrix(x, sigma, myu, CV_COVAR_NORMAL | CV_COVAR_COLS, CV_32F);
				if (x.rows != myu.rows || x.rows != sigma.rows)
					std::cout << "Something bad happened as the vectors x and myu and the matrix sigma does not match.";
				float scalar = 1 / (pow(2 * M_PI, d / 2)* pow(d, 0.5f));
				
				cv::Mat difference;
				cv::subtract(x, myu, difference, cv::Mat(), CV_32F);
				cv::Mat exponent = -0.5f * difference.t() * sigma.inv() * difference;

				float Probability = scalar * exp(exponent.at<float>(0, 0));
				ProbabilityForClass[i] = Probability;
			}

			int MostProbableClass = -1;
			float maxProbability = 0;
			for (size_t i = 0; i < ProbabilityForClass.size(); i++) {
				float currentClassProbability = ProbabilityForClass[i];
				if (currentClassProbability > maxProbability) {
					maxProbability = currentClassProbability;
					MostProbableClass = i;
				}
			}

			ClassificationMask.at<uchar>(y_t, x_t) = MostProbableClass;
		}
	}
	return ClassificationMask;
}


void StartAssignment2()
{
	cv::Mat TrainingMask = ReadImageFromTXT("training_mask.txt");

	cv::Mat TrainingImg = ReadImageFromTXT("mosaic1_train.txt");
	std::vector<int> offsets;
	//Direction 1
	offsets.push_back(1);
	offsets.push_back(0);
	//Direction 2
	offsets.push_back(0);
	offsets.push_back(1);

	//Training
	//Calculate the features, Q1 to Qn.
	std::vector<cv::Mat> QFeatImgs = ComputeQFeatureImgs(TrainingImg, offsets);
	
	//Use the test mask to generate a descriptor using the N means of the N features for all M classes. We should have M descriptors now.
	std::vector<ClassDescriptor> Descriptors = ComputeClassDescriptors(QFeatImgs, TrainingMask);
	//Use the function cv::calcCovarMatrix() to calculate the covariance matrix for each class descriptor.
	

	//Classification
	cv::Mat ImgTOClassify = ReadImageFromTXT("mosaic2_test.txt");

	//Calculate the new feature imgs.
	QFeatImgs = ComputeQFeatureImgs(ImgTOClassify, offsets);

	std::vector<cv::Mat> CovarMats;
	//Generate feature imgs. Store each pixel into an array called X;
	//Compute the QFeature
	//Calculate descriptor from the QFeature images.
	//Use the function given in the book to calculate the probability of a pixel being class 1...M. Select the one with the highest probability. 
	//Input is the image, the class descriptors and the covariance matrices.
	auto samples = computeSamples(TrainingMask, QFeatImgs);
	cv::Mat classifiedImage = MultivariateGaussian(ImgTOClassify, Descriptors, samples, QFeatImgs);

	cv::imshow("ClassifiedImage", classifiedImage);
	float accuracy = ConfusionMatrix(TrainingMask, classifiedImage);

	std::cout << "Accuracy of classifier: " << accuracy;

}

float ComputeQFeature(cv::Mat GLCM, size_t start_x_t, size_t end_x_t, size_t start_y_t, size_t end_y_t)
{
	// ReSharper disable once CppInitializedValueIsAlwaysRewritten
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
std::vector<float> ComputeQFeatures(std::vector<cv::Mat> GLCMs)
{
	std::vector<float> Qs;
	for (size_t i = 0; i < GLCMs.size(); i++) {
		cv::Mat GLCM = GLCMs[i];
		size_t width_t = GLCM.cols;
		size_t height_t = GLCM.rows;
		size_t wStep = width_t / 2;
		size_t hStep = height_t / 2;
		int qIndex = 0;
		for (size_t u = 0; u < 2; u++) {
			for (size_t j = 0; j < 2; j++) {
				if (qIndex == 2)
				{
					qIndex++;
					continue;
				}

				size_t startX = wStep*j, endX = wStep + wStep*j, startY = hStep*u, endY = hStep + hStep*u;

				float Q = ComputeQFeature(GLCM, startX, endX, startY, endY);
				Qs.push_back(Q);
				qIndex++;
			}
		}
	}
	return Qs;
}

std::vector<cv::Mat> ComputeGLCM(const cv::Mat& Img, std::vector<int> XYOffsets)
{
	int TotalMeasurements = 0;
	cv::Mat dst(Img.cols, Img.rows, CV_8UC1);
	cv::resize(Img, dst, cv::Size(dst.cols, dst.rows), 0, 0, CV_INTER_CUBIC);
	int NumGrayLevels = 16;
	ReduceGrayLevels(dst, NumGrayLevels);

	//Create and zero out matrix. 
	std::vector<cv::Mat> GLCMs;
	for (size_t i = 0; i < XYOffsets.size()/2; i++) {
		GLCMs.push_back(cv::Mat::zeros(NumGrayLevels, NumGrayLevels, CV_32F));
	}
	
	//For each direction, create a GLCM. It is not normalized. It is symetrical.
	for (size_t i = 0; i < XYOffsets.size(); i += 2) {
		for (int y = 0; y < dst.rows - XYOffsets[i + 1]; y++) {
			for (int x = 0; x < dst.cols - XYOffsets[i]; x++) {
				int deltaX = XYOffsets[i];
				int deltaY = XYOffsets[i + 1];
				size_t GrayLevelBase = dst.at<uchar>(cv::Point(y, x));
				size_t GrayLevelTarget = dst.at<uchar>(cv::Point(y + deltaY, x + deltaX));

				GLCMs[i/2].at<float>(GrayLevelBase, GrayLevelTarget) += 1;
				//GLCMs[i/2].at<float>(GrayLevelTarget, GrayLevelBase) += 1;
				TotalMeasurements = TotalMeasurements + 1;
			}
		}
	}

	return GLCMs;
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
			size_t currentClassID = mask.at<uchar>(y_t, x_t);
			if (currentClassID > numClasses) numClasses = currentClassID;
		}
	}

	
	//Create the descriptors.
	for (size_t i = 0; i < numClasses+1; i++)
	{
		descriptors.push_back(ClassDescriptor(featureImgs.size(), i));
	}

	//Compute the class avarages.
	//For every pixel.
	for (size_t y_t = 0; y_t < mask.rows; y_t++) {
		for (size_t x_t = 0; x_t < mask.cols; x_t++) {

			int classID = mask.at<uchar>(y_t, x_t);
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

	for (size_t i = 0; i < descriptors.size(); i++) {
		for (size_t j_t = 0; j_t < descriptors[i].Descriptor.rows; j_t++) {
			std::cout << ", Descriptor " << i << ": mean " << j_t << ": " << descriptors[i].Descriptor.at<float>(j_t, 0);
		}
		std::cout << std::endl;
	}

	return descriptors;
}

std::vector<cv::Mat> ComputeQFeatureImgs(cv::Mat& Img, std::vector<int> XYOffsets)
{
	std::vector<cv::Mat> QFeatureIms;
	for (size_t i = 0; i < (XYOffsets.size() / 2) * 3; i++) {
		QFeatureIms.push_back(cv::Mat(Img.rows, Img.cols, CV_32F));
	}
	for (size_t i = 0; i < QFeatureIms.size(); i++)
	{
		for (size_t y_t = 0; y_t < QFeatureIms[i].rows; y_t++)
		{
			for (size_t x_t = 0; x_t < QFeatureIms[i].cols; x_t++)
			{
				QFeatureIms[i].at<float>(y_t, x_t) = 0;
			}
		}
	}

	int WindowSize = 31;


	for (size_t y = 0; y < Img.rows - WindowSize; y++) {
		std::cout << "Pos: " << y << std::endl;
		for (size_t x = 0; x < Img.cols - WindowSize; x++) {
			cv::Mat Window = Img(cv::Rect(y, x, WindowSize, WindowSize));
			std::vector<cv::Mat> GLCMs = ComputeGLCM(Window, XYOffsets);

			//Compute the QFeatures for thew 4 parts.
			std::vector<float> QFeats = ComputeQFeatures(GLCMs);

			//For every computed Q feature set the pixel on the matching Q feature image.
			for (size_t i = 0; i < QFeatureIms.size(); i++) {
				float value = QFeats[i];
				cv::Mat currentFeatureImg = QFeatureIms[i];
				currentFeatureImg.at<float>(y + 1 + (WindowSize / 2), x + 1 + (WindowSize / 2)) = value;
			}
		}
	}
	
	return QFeatureIms;
}

cv::Mat NormalizeMat(cv::Mat& mat)
{
	cv::Mat returnmat(mat.rows, mat.cols, CV_8UC1);
	float max = 0;
	for (size_t y = 0; y < mat.rows; y++) {
		for (size_t x = 0; x < mat.cols; x++) {
			if (max < mat.at<float>(y, x)) max = mat.at<float>(y, x);
		}
	}

	for (size_t y = 0; y < mat.rows; y++) {
		for (size_t x = 0; x < mat.cols; x++) {
			float oldVal = mat.at<float>(y, x);
			if (oldVal != 0.0f)
				std::cout << "";
			float newVal = (oldVal / max);
			newVal = newVal * 255;
			returnmat.at<uchar>(y, x) = (uchar)newVal;
		}
	}

	return returnmat;
}

cv::Mat getPixelDescriptor(std::vector<cv::Mat>& featureImgs, size_t y, size_t x)
{
	int d = featureImgs.size();
	cv::Mat descriptor(d, 1, CV_32F);

	for (size_t i = 0; i < d; i++) {
		descriptor.at<float>(i, 0) = featureImgs[i].at<float>(y, x);
	}

	return descriptor;
}

std::vector<std::vector<std::vector<float>>> computeSamples(cv::Mat& mask, std::vector<cv::Mat>& featureImgs)
{
	std::vector<std::vector<std::vector<float>>> samples;

	std::vector<std::vector<float>> samplesForClass(featureImgs.size());

	//For every class
	int numClasses = 5;
	for (size_t i = 0; i < numClasses; i++)
	{
		samplesForClass.clear();
		samplesForClass.resize(featureImgs.size());
		//Find every pixel for that class.
		for (size_t y = 0; y < mask.rows; y++)
		{
			for (size_t x = 0; x < mask.cols; x++)
			{
				if (mask.at<uchar>(y, x) == i) {
					//Add the value of each feature image at that pixel.
					for (size_t feat_index_t = 0; feat_index_t < featureImgs.size(); feat_index_t++) {
						samplesForClass[0].push_back(featureImgs[feat_index_t].at<float>(y, x));
					}
				}
			}
		}
		samples.push_back(samplesForClass);
	}

	return samples;
}