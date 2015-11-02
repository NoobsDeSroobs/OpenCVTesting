#include "ImageReader.h"
#include <fstream>

cv::Mat ReadImageFromTXT(std::string path)
{
	cv::Mat Image(5000, 5000, CV_32F);
	std::ifstream backstory("backstory.txt");
	size_t x = 0;
	size_t y = 0;

	if (backstory.is_open()) {
		while (backstory.good()) {
			std::string line;
			getline(backstory, line);
			size_t pos = 0;
			std::string token;
			std::string delimiter = ",";

			while ((pos = line.find(delimiter)) != std::string::npos) {
				token = line.substr(0, pos);
				float value = std::stoi(token);
				Image.at<float>(x, y) = value;
				x++;
				std::cout << token << std::endl;
				line.erase(0, pos + delimiter.length());
			}
			y++;
			x = 0;
		}

		//This will, hopefully, reduce the size to the image size.
		Image.resize(y);
		Image.colRange(0, x);
		
		backstory.close();
	} else {
		std::cout << "Unable to open file" << std::endl << std::endl;
	}

	return Image;
}