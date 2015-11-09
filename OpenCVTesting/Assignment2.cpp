#include "Assignment2.h"
#include "ImageReader.h"
#define _USE_MATH_DEFINES
#include <math.h>

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


cv::Mat MultivariateGaussian(cv::Mat Img, std::vector<ClassDescriptor> Descriptors, std::vector<cv::Mat> CovarMats)
{

	float d = Descriptors.size();
	cv::Mat ClassificationMask(Img.rows, Img.cols, CV_8U);
	for (size_t y_t = 0; y_t < Img.rows; y_t++) {
		for (size_t x_t = 0; x_t < Img.cols; x_t++) {

			std::vector<float> ProbabilityForClass(d);
			//Calculate the probability for all the classes. 
			for (size_t i = 0; i < d + 1; i++) {
				cv::Mat x(d, 1, CV_32F);
				//FeatureImgs.at<float>(0, 0);
				cv::Mat sigma = CovarMats[0];
				float scalar = 1 / (pow(2 * M_PI, d / 2)* pow(d, 0.5f));
				cv::Mat exponent = -0.5f * (x - Descriptors[0].Descriptor) * sigma.inv() * (x - Descriptors[0].Descriptor);

				float Probability = scalar* exp(exponent.at<float>(0, 0));
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
	/*
	std::vector<cv::Mat> GLCM;
	GLCM.push_back(ReadImageFromTXT("texture1dx0dy1.txt"));
	GLCM.push_back(ReadImageFromTXT("texture1dx1dy0.txt"));
	GLCM.push_back(ReadImageFromTXT("texture1dx1dy-1.txt"));
	GLCM.push_back(ReadImageFromTXT("texture1dx-1dy1.txt"));

	GLCM.push_back(ReadImageFromTXT("texture2dx0dy1.txt"));
	GLCM.push_back(ReadImageFromTXT("texture2dx1dy0.txt"));
	GLCM.push_back(ReadImageFromTXT("texture2dx1dy-1.txt"));
	GLCM.push_back(ReadImageFromTXT("texture2dx-1dy1.txt"));

	GLCM.push_back(ReadImageFromTXT("texture3dx0dy1.txt"));
	GLCM.push_back(ReadImageFromTXT("texture3dx1dy0.txt"));
	GLCM.push_back(ReadImageFromTXT("texture3dx1dy-1.txt"));
	GLCM.push_back(ReadImageFromTXT("texture3dx-1dy1.txt"));

	GLCM.push_back(ReadImageFromTXT("texture4dx0dy1.txt"));
	GLCM.push_back(ReadImageFromTXT("texture4dx1dy0.txt"));
	GLCM.push_back(ReadImageFromTXT("texture4dx1dy-1.txt"));
	GLCM.push_back(ReadImageFromTXT("texture4dx-1dy1.txt"));
	
		
	bool success;
	success = cv::imwrite("Images/texture1dx0dy1.png" , GLCM[0]);
	success = cv::imwrite("Images/texture1dx1dy0.png" , GLCM[1]);
	success = cv::imwrite("Images/texture1dx1dy-1.png", GLCM[2]);
	success = cv::imwrite("Images/texture1dx-1dy1.png", GLCM[3]);
								 
	success = cv::imwrite("Images/texture2dx0dy1.png" , GLCM[4]);
	success = cv::imwrite("Images/texture2dx1dy0.png" , GLCM[5]);
	success = cv::imwrite("Images/texture2dx1dy-1.png", GLCM[6]);
	success = cv::imwrite("Images/texture2dx-1dy1.png", GLCM[7]);
							 
	success = cv::imwrite("Images/texture3dx0dy1.png" , GLCM[8]);
	success = cv::imwrite("Images/texture3dx1dy0.png" , GLCM[9]);
	success = cv::imwrite("Images/texture3dx1dy-1.png", GLCM[10]);
	success = cv::imwrite("Images/texture3dx-1dy1.png", GLCM[11]);
							 
	success = cv::imwrite("Images/texture4dx0dy1.png" , GLCM[12]);
	success = cv::imwrite("Images/texture4dx1dy0.png" , GLCM[13]);
	success = cv::imwrite("Images/texture4dx1dy-1.png", GLCM[14]);
	success = cv::imwrite("Images/texture4dx-1dy1.png", GLCM[15]);
	*/
	

	cv::Mat TrainingMask = ReadImageFromTXT("training_mask.txt");

	cv::Mat TrainingImg = ReadImageFromTXT("mosaic1_train.txt");
	std::vector<int> offsets;
	//Direction 1
	offsets.push_back(1);
	offsets.push_back(0);
	//Direction 2
	offsets.push_back(0);
	offsets.push_back(1);

	//Img = GLCM1[0];

	//Training
	//Calculate the features, Q1 to Qn.
	std::vector<cv::Mat> QFeatImgs = ComputeQFeatureImgs(TrainingImg, offsets);

	for (size_t i = 0; i < QFeatImgs.size(); i++) {
		std::cout << QFeatImgs[i].at<float>(0, 0);
		std::cout << QFeatImgs[i].at<float>(300, 250);
		cv::imshow("FeatureImage", QFeatImgs[i]);
		cv::waitKey();
	}
	
	//Use the test mask to generate a descriptor using the N means of the N features for all M classes. We should have M descriptors now.
	std::vector<ClassDescriptor> Descriptors = ComputeClassDescriptors(QFeatImgs, TrainingMask);
	//Use the function cv::calcCovarMatrix() to calculate the covariance matrix for each class descriptor.
	

	//Classification
	cv::Mat ImgTOClassify = ReadImageFromTXT("mosaic2_test.txt");

	//Calculate the new feature imgs.
	QFeatImgs = ComputeQFeatureImgs(TrainingImg, offsets);

	for (size_t i = 0; i < QFeatImgs.size(); i++) {
		std::cout << QFeatImgs[i].at<float>(0, 0);
		std::cout << QFeatImgs[i].at<float>(300, 250);
		cv::imshow("FeatureImage", QFeatImgs[i]);
		cv::waitKey();
	}

	std::vector<cv::Mat> CovarMats;
	for (size_t i = 0; i < Descriptors.size(); i++) {
		CovarMats.push_back(cv::Mat(Descriptors.size(), Descriptors.size(), CV_32F));
	}
	
	for (size_t i = 0; i < Descriptors.size(); i++) {
		cv::calcCovarMatrix(QFeatImgs, CovarMats[i], Descriptors[i].Descriptor, CV_COVAR_NORMAL);
	}
	//Generate feature imgs. Store each pixel into an array called X;
	//Compute the QFeature
	//Calculate descriptor from the QFeature images.
	//Use the function given in the book to calculate the probability of a pixel being class 1...M. Select the one with the highest probability. 
	//Input is the image, the class descriptors and the covariance matrices.
	cv::Mat classifiedImage = MultivariateGaussian(ImgTOClassify, Descriptors, CovarMats);

	float accuracy = ConfusionMatrix(TrainingMask, classifiedImage);

	std::cout << "Accuracy of classifier: " << accuracy;

}

float ComputeQFeature(cv::Mat GLCM, size_t start_x_t, size_t end_x_t, size_t start_y_t, size_t end_y_t)
{
	//std::cout << "X: " << start_x_t << " - " << end_x_t << " Y: " << start_y_t << " - " << end_y_t << std::endl;
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

		for (size_t i = 0; i < 2; i++) {
			for (size_t j = 0; j < 2; j++) {
				size_t startX = wStep*j, endX = wStep + wStep*j, startY = hStep*i, endY = hStep + hStep*i;

				float Q = ComputeQFeature(GLCM, startX, endX, startY, endY);
				Qs.push_back(Q);
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
				GLCMs[i/2].at<float>(GrayLevelTarget, GrayLevelBase) += 1;
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

			/*for (size_t i = 0; i < featureImgs.size(); i++) {
					std::cout << " Feat img " << i << ": " << featureImgs[i].at<float>(y_t, x_t);
			}
			std::cout << "\nDescriptor feat" << std::endl;*/

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
	//TODO Figure out how to determine the size of this vector.
	std::vector<cv::Mat> QFeatureIms;
	for (size_t i = 0; i < (XYOffsets.size() / 2) * 4; i++) {
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

			/*for (size_t i = 0; i < GLCMs.size(); i++) {
				std::string index = std::to_string(i);
				std::string WinNameBase = "GLCM";
				std::string WinName = WinNameBase.append(index);
				cv::imshow(WinName, GLCMs[i]);
			}
			cv::waitKey();*/

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

void NormalizeMat(cv::Mat& mat)
{
	float max = 0;
	for (size_t y = 0; y < mat.rows; y++) {
		for (size_t x = 0; x < mat.cols; x++) {
			if (max < mat.at<float>(y, x)) max = mat.at<float>(y, x);
		}
	}

	for (size_t y = 0; y < mat.rows; y++) {
		for (size_t x = 0; x < mat.cols; x++) {
			mat.at<float>(y, x) = (mat.at<float>(y, x) / max) * 255;
		}
	}


}