#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <iostream>
#include <random>

#include <opencv2\core\mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "WBSSettings.h"
#include "WBSStructs.h"
#include "StrongClassifier.h"

class CPUObjectDetector
{
public:
	CPUObjectDetector(int windowSize, int windowStep, float pyramidStep, bool measurePerformance);
	~CPUObjectDetector();

	void detectObjectInImage(std::string imgPath);

	void setStrongClassifier(StrongClassifier* s);


private:
	const int slidingWindowSize;
	const int slidingWindowStep;
	const float pyramidStep;
	std::vector<cv::Mat>::iterator pyramidIterator;
	bool measurePerformance;
	
	// random features
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution;
	// real strong classifier
	StrongClassifier* trainedClassifier;
	
	// vector with rectangle around founded objects
	std::vector<std::pair<cv::Rect, cv::Size>> foundedObjects;

	cv::Mat inputImg;
	std::vector<cv::Mat> imagePyramid;
	std::pair<int, int> slidingWindowPosition;
	cv::Mat processedImg;
	cv::Size processedImgSize;

	void initSlidingWindowPosition();
	void setNextSlidingWindowPosition();
	void getNextImageFromPyramid();
	void setNextImage(cv::Mat *i);
	//! Private function of CPUObjectDetector class preprocessImage to create pyramid for detection.
	/*!
	\param img a grayscale image in cv::Mat
	*/
	void preprocessImage();
	void subSampleInputImage();

	void initForNextInputImage();

	// apply strong classifier on processed image for current sliding window
	void applyStrongClassifier();
	void addFoundedObject();

	void showImageWithFoundedObjects();
};

