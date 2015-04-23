#pragma once
#include <opencv2\opencv.hpp>

class Layer
{
public:
	Layer() : XSize(0), YSize(0), visible(true) {};
	Layer(cv::Mat l) : layer(l), visible(true) {};
	Layer(size_t x, size_t y) : XSize(x), YSize(y), visible(true) {};


	cv::Mat layer;
	size_t XSize;
	size_t YSize;

	bool visible;
};