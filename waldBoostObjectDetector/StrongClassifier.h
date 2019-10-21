#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include "opencv2\core\mat.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "HaarlikeFeatureGenerator.h"
#include "WBSStructs.h"
#include "WBSSettings.h"



class StrongClassifier
{
public:
	StrongClassifier(int size, int maxLimitWeak);
	~StrongClassifier();

	void prepareSet(std::string setPath, bool isPositive);
	void setPositiveSetPath(std::string positivePath = POSITIVE_SET_PATH);
	void setNegativeSetPath(std::string negativePath = NEGATIVE_SET_PATH);

	//! With strong class parameters set, the function find the most succesfull weak classifiers.
	/*! \param falsePositiveRate is desired final false positive rate
	 *! \param falseNegaticeRate is desired final false negative rate
	*/
	void learModel(float falsePositiveRate, float falseNegativeRate);

	std::vector<cv::Rect> predict(cv::Mat());

	//! Function cut image into parts of size defined by StrongClassifier Class. It will not copy data!
	std::vector<cv::Mat>  prepareImage(cv::Mat input);
	
	std::vector<std::pair
		<wbs::HaarlikeFeature*, std::pair<float, float>>
	> getWeakClassifiersLine() {
		return weakClassifiersLine;
	}

private:
	std::string positiveSetPath;
	std::string negativeSetPath;
	int positiveSetSize;
	int negativeSetSize;

	int imageSize;
	HaarlikeFeatureGenerator* haarGen;
	int maxLimitOfWeakClassifiers;

	float thresholdNegative;
	float thresholdPositive;

	std::pair<wbs::HaarlikeFeature*, std::pair<float, float>> possibleNewWeakClassifier;

	// variable to store list of weak classifiers with their decision boundaries
	std::vector<std::pair
		<wbs::HaarlikeFeature*, std::pair<float, float>>
		> weakClassifiersLine;

	//! Apply Haarlike feature to given image, if 
	int applyFeature(wbs::HaarlikeFeature* f, std::vector<cv::Mat> i, bool isPositive);
	int applyFeature(std::pair<wbs::HaarlikeFeature*, std::pair<float, float>>, std::vector<cv::Mat>*, bool isPositive);

	//! Calculate Horizontal feature and return 0 if the found category is right, otherwise return 1.
	int applyHorizontalEdgeFeature(wbs::HaarlikeFeature* f, cv::Mat i, bool isPositive, float thresNeg = -1.0f, float thresPos = -1.0f);

	//! Check if current Haatlike feature is better than previous best one and replace if not already in Classifier.
	void checkIfBetterFeatureAndCopy(wbs::HaarlikeFeature*, int, int);

	//! Check if current Haarlike feature is already in strong classifier, we do not want to have it twice there.
	bool weakClassifiersLineContain(wbs::HaarlikeFeature*);
	bool areaSame(wbs::HaarlikeFeatureAreas*, wbs::HaarlikeFeatureAreas*);

	void addNewWeakClassifier();

	//! Apply current strong classificator to given image set and return likelihood
	float applyStrongClassificator(std::vector<cv::Mat>*, bool);
};

