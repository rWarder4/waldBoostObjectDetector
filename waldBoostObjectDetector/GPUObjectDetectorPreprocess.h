#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <iostream>
#include <random>
#include <cuda.h>
#include <math.h>

#include <opencv2\core\mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "WBSSettings.h"
#include "WBSStructs.h"
#include "StrongClassifier.h"

extern int GPUObjectDetector(double* imageData, int* imageDataDescriptor, int imageDataDescriptorStep, int imageDataSize, double* weakClassProb, int weakClassNum);

class GPUObjectDetectorPreprocess
{
public:
	GPUObjectDetectorPreprocess(int windowSize, int windowStep, float pyrStep, bool measurePerformance);
	~GPUObjectDetectorPreprocess();

	void detectObjectInImage(std::string imgPath);

private:
	const int slidingWindowSize;
	const int slidingWindowStep;
	const float pyramidStep;
	bool measurePerformance;

	std::string inputImagePath;
	int numOfImages;

	cv::Mat imagePyramidMat;
	std::vector<double> imagePyramid;

	double* imagePyramidArray;
	int imagePyramidArraySize;
	int* imagePyramidArrayDescriptor;
	int imagePyramidArrayDescriptorStep;

	// random features
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution;
	double* weakClassifierProbabilities;
	int numOfWeakClassifiers;

	// real strong classifier
	StrongClassifier* trainedClassifier;

	
	//! Preprocess Input image to create pyramid of images for detection.
	/*!
	\param img a grayscale image in cv::Mat
	*/
	void preprocessImage(cv::Mat *inputImg);

	void subSampleInputImage(cv::Mat* i);
	void buildPyramid(cv::Mat* inputImg);
	cv::Mat createMask(cv::Size pyrSize, cv::Rect);

	void generateProbabilitiesForWeakClassifiers();
};

