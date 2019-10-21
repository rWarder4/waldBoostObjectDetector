#include "StrongClassifier.h"



StrongClassifier::StrongClassifier(int size, int maxLimitWeak)
{
	imageSize = size;
	maxLimitOfWeakClassifiers = maxLimitWeak;
	haarGen = new HaarlikeFeatureGenerator(size);
	positiveSetSize = 0;
	negativeSetSize = 0;
	// init possible weak classifier
	possibleNewWeakClassifier.first = NULL;
	possibleNewWeakClassifier.second.first = -1;
	possibleNewWeakClassifier.second.second = -1;
}


StrongClassifier::~StrongClassifier()
{
}

void StrongClassifier::prepareSet(std::string path, bool isPositive)
{
	int imgNumber = 0;
	for (auto & p : std::experimental::filesystem::directory_iterator(path)) {
		//std::cout << p << std::endl;
		cv::Mat img = cv::imread(p.path().string(), CV_LOAD_IMAGE_GRAYSCALE);
		
		// resize to image_width
		int width = img.cols;
		int height = img.rows;
		float aspectRatio = float(width) / float(height);
		//std::cout << width << ", " << height << " | " << aspectRatio << std::endl;
		int newWidth = IMAGE_WIDTH;
		int newHeight = float(newWidth) / aspectRatio;
		//std::cout << newWidth << ", " << newHeight << " | " << aspectRatio << std::endl;
		
		// we have new sizes, resize image, set new name and save
		cv::resize(img, img, cv::Size(newWidth, newHeight));

		// check if dimension are divisible by 24, if not edit
		if (img.rows%imageSize != 0) {
			int divH = img.rows / imageSize;
			int newDivH = imageSize * divH;
			cv::Mat cropImg = cv::Mat(img, cv::Rect(0, 0, img.cols, newDivH));
			img = cropImg;
		}

		// cut image into parts which will be processed separately
		std::vector<cv::Mat> imageSegmented = prepareImage(img);

		for (auto & value : imageSegmented) {
			std::string newFileName = "";
			isPositive ? newFileName += POSITIVE_SET_PATH : newFileName += NEGATIVE_SET_PATH;
			newFileName += "sample_" + std::to_string(imgNumber) + ".jpg";
			std::cout << newFileName << std::endl;
			cv::imwrite(newFileName, value);
			imgNumber++;
		}

		isPositive ? positiveSetSize = imgNumber : negativeSetSize = imgNumber;
	}
}

void StrongClassifier::setPositiveSetPath(std::string positivePath)
{
	positiveSetPath = positivePath;
}

void StrongClassifier::setNegativeSetPath(std::string negativePath)
{
	negativeSetPath = negativePath;
}

void StrongClassifier::learModel(float falsePositiveRate, float falseNegativeRate)
{
	// variable for current Haarlike feature
	wbs::HaarlikeFeature* tempFeature;

	// load all images into memory
	std::vector<cv::Mat> positiveSet;
	std::vector<cv::Mat> negativeSet;
	for (auto & p : std::experimental::filesystem::directory_iterator(positiveSetPath)) {
		cv::Mat img = cv::imread(p.path().string(), CV_LOAD_IMAGE_GRAYSCALE);
		// calculate integral image
		img.convertTo(img, CV_32F);
		cv::Mat integralImg;
		cv::integral(img, integralImg);
		positiveSet.push_back(integralImg);
	}
	// back up positiveSet
	std::vector<cv::Mat> positiveSetBackUp = positiveSet;

	// go through all image in negative
	for (auto & p : std::experimental::filesystem::directory_iterator(negativeSetPath)) {
		//std::cout << p << std::endl;
		cv::Mat img = cv::imread(p.path().string(), CV_LOAD_IMAGE_GRAYSCALE);
	}

	std::cout << positiveSet.size() << std::endl;

	// Dont have data, stop.
	if (positiveSet.size() <= 0 && negativeSet.size() <= 0) {
		return;
	}

	//-------------------- START OF LEARNING ALGORITHM --------------------------
	// 0. INIT WEIGHTS and SET CLASS DEFINITIONS
	float weight = 1 / positiveSet.size();
	thresholdNegative = (1 - falsePositiveRate) / falseNegativeRate;
	thresholdPositive = falsePositiveRate / (1 - falseNegativeRate);

	// REPEATE UNTIL WE FOUND ENOUGH BEST WEAK CLASSIFIERS
	while (weakClassifiersLine.size() != maxLimitOfWeakClassifiers) {
		std::cout << "Number of weak classifiers: " << weakClassifiersLine.size() << std::endl;
		// 1. FIND OUT BEST WEAK CLASSIFIER
		// get first Haar like feature
		tempFeature = haarGen->getNextFeature();
		// iterate over all available Haarlike Features
		while (tempFeature->xPos != -1) {
			// go through all avaiable images and get the score, save the one with least error which wasn't used yet
			// apply on positive set 
			int positiveErrors = applyFeature(tempFeature, positiveSet, true);
			int negativeErrors = applyFeature(tempFeature, negativeSet, false);
			// check if this Haarlike feature scored better than previous best
			checkIfBetterFeatureAndCopy(tempFeature, positiveErrors, negativeErrors);
			// get next Haarlike feature and apply on all segments of image
			tempFeature = haarGen->getNextFeature();
			if (tempFeature->xPos == -1)
				break;
		}
		// we iterate over all available Haarlike features and found the one with least errors
		addNewWeakClassifier();
		// reset Haarlike feature generator for next search
		haarGen->resetHaarlikeGenerator();
		// 2. APPLY CURRENT STRONG CLASSIFIER TO SET AND ESTIMATE LIKELIHOOD RATIO
		float alfa, beta;
		positiveSet.size()>0 ? beta = applyStrongClassificator(&positiveSet, true) : beta = falseNegativeRate;
		negativeSet.size()>0 ? alfa = applyStrongClassificator(&negativeSet, false) : alfa = falsePositiveRate;
		// 3. FIND THRESHOLDS FOR CURRENT STRONG CLASSIFIER STATE AND UPDATE THEM
		thresholdNegative = (1 - beta) / alfa;
		thresholdPositive = beta / (1 - alfa);

		// 4. THROW AWAY SAMPLES WHICH ARE ALREADY WELL RECONGIZED

		// 5. SAMPLA NEW DATA TO TRAINING SET IF POSSIBLE

		// check if we have any data to train on push them back again
		if (positiveSet.size() <= 0 && negativeSet.size() <= 0) {
			//positiveSet = positiveSetBackUp;
			break;
		}
	}
}

//! Cut image into parts of set size.
std::vector<cv::Mat> StrongClassifier::prepareImage(cv::Mat input)
{
	std::vector<cv::Mat> output;
	for (int y = 0; y < input.rows; y += imageSize)
	{
		for (int x = 0; x < input.cols; x += imageSize)
		{
			cv::Rect rect = cv::Rect(x, y, imageSize, imageSize);
			output.push_back(cv::Mat(input, rect));
		}
	}
	return output;
}

//! Apply Haatlike feature on all image in vector i and return number of incorrectly classified images. Return -1 if empty parameters given.
int StrongClassifier::applyFeature(wbs::HaarlikeFeature *f, std::vector<cv::Mat> i, bool isPositive)
{
	int errorScore = 0;
	// check if image set is not empty
	if (i.size() <= 0) {
		return -1;
	}
	// for all image check score
	for (auto & img : i) {
		switch (f->type) {
		case wbs::HORIZONTAL_EDGE:
				int score = applyHorizontalEdgeFeature(f, img, isPositive);
				if (score == 1) {
					errorScore++;
				}
				break;
		}
	}

	return errorScore;
}

int StrongClassifier::applyFeature(std::pair<wbs::HaarlikeFeature*, std::pair<float, float>> f, std::vector<cv::Mat>* i, bool isPositive)
{
	int rightScore = 0;
	// check if image set is not empty
	if (i->size() <= 0) {
		return -1;
	}
	std::vector<cv::Mat>::iterator it = i->begin();

	// for all image check score
	while (it != i->end()) {
		switch (f.first->type) {
			case wbs::HORIZONTAL_EDGE:
				int score = applyHorizontalEdgeFeature(f.first, *it ,isPositive, f.second.first, f.second.second);
				if (score == 0) {
					rightScore++;
					// remove image from set already classified
					it = i->erase(it);
				}
				else {
					++it;
				}
				break;
		}
	}

	return rightScore;
}

//! Calculate Horizontal feature and return 0 if the found category is right, otherwise return 1.
/*! \param f is current Haarlike feature
 *  \param i is given image
 *  \param isPositive is set to TRUE if given image is from positive data set
 *  \param thresNeg is used for object detection and it contains Negative threshold for given Haarlike feature
 *  \param thresPos is used for object detection and it contains Positive threshold for given Haarlike feature
*/
int StrongClassifier::applyHorizontalEdgeFeature(wbs::HaarlikeFeature * f, cv::Mat i, bool isPositive, float thresNeg, float thresPos)
{
	// get current class borders
	float tN;
	thresNeg == -1 ? tN = thresholdNegative : tN = thresNeg;
	float tP;
	thresPos == -1 ? tP = thresholdPositive : tP = thresPos;

	// evaluate current img with current Haarlike feature
	double temp001 = i.at<double>(f->xPos + f->a1->x1, f->yPos + f->a1->y1);
	double area1Value = i.at<double>(f->xPos + f->a1->x1, f->yPos + f->a1->y1) +
		i.at<double>(f->xPos + f->a1->x2, f->yPos + f->a1->y2) -
		(i.at<double>(f->xPos + f->a1->x2, f->yPos + f->a1->y1) +
			i.at<double>(f->xPos + f->a1->x1, f->yPos + f->a1->y2));

	double area2Value = i.at<double>(f->xPos + f->a2->x1, f->yPos + f->a2->y1) +
		i.at<double>(f->xPos + f->a2->x2, f->yPos + f->a2->y2) -
		(i.at<double>(f->xPos + f->a2->x2, f->yPos + f->a2->y1) +
			i.at<double>(f->xPos + f->a2->x1, f->yPos + f->a2->y2));

	double score = area2Value - area1Value;

	// correct Classification
	if (((double)score >= tN && isPositive) ||
		((double)score <= tP && !isPositive) ) {
		return 0;
	}
	// is Positive but we determine it's negative -> wrong Classification, return error + 1
	else if (((double)score <= tP && isPositive) ||
		((double)score >= tN && !isPositive) ){
		return 1;
	}
	// this weak classification cannot determine if is Positive or isn't
	else {
		return -1;
	}
	return -1;
}


//! Check if current Haatlike feature is better than previous best one and replace if not already in Classifier.
/*! \param f
*	\param positiveErrors is number of errors on positive set
*	\param negativeErrors is number of errors on negative set
*/
void StrongClassifier::checkIfBetterFeatureAndCopy(wbs::HaarlikeFeature *f, int positiveErrors, int negativeErrors)
{
	// check if the current Haarlike feature is already in strong classifier -> skip
	if (weakClassifiersLineContain(f))
		return;

	if (possibleNewWeakClassifier.first == NULL || (possibleNewWeakClassifier.second.first + possibleNewWeakClassifier.second.second > positiveErrors + negativeErrors)) {
		possibleNewWeakClassifier.first = haarGen->copyFeature(f);
		possibleNewWeakClassifier.second.first = positiveErrors;
		possibleNewWeakClassifier.second.second = negativeErrors;
	}
}

//! Check if current Haarlike feature is already in strong classifier, we do not want to have it twice there.
bool StrongClassifier::weakClassifiersLineContain(wbs::HaarlikeFeature * f)
{
	if (weakClassifiersLine.size() <= 0) {
		return false;
	}
	wbs::HaarlikeFeature * featureIn;
	for (auto & item : weakClassifiersLine) {
		featureIn = item.first;
		if (featureIn->type != f->type) { // different type
			return false;
		}
		else if (featureIn->xPos != f->xPos || featureIn->yPos != f->yPos) { // different position
			return false;
		}
		else if (!areaSame(featureIn->a1, f->a1) || !areaSame(featureIn->a2, f->a2) || !areaSame(featureIn->a3, f->a3) || !areaSame(featureIn->a4, f->a4)) {
			return false;
		}
	}

	return true;
}

bool StrongClassifier::areaSame(wbs::HaarlikeFeatureAreas*area1, wbs::HaarlikeFeatureAreas* area2)
{
	if (area1->x1 != area2->x1 || area1->x2 != area2->x2 || area1->y1 != area2->y1 || area1->y2 != area2->y2) {
		return false;
	}

	return true;
}

void StrongClassifier::addNewWeakClassifier()
{
	std::pair<wbs::HaarlikeFeature*, std::pair<float, float>> newWeakClass;
	newWeakClass.first = possibleNewWeakClassifier.first;
	newWeakClass.second.first = thresholdNegative;
	newWeakClass.second.second = thresholdPositive;

	// reset possible new weak classifier variable
	possibleNewWeakClassifier.first = NULL;
	possibleNewWeakClassifier.second.first = -1;
	possibleNewWeakClassifier.second.second = -1;

	weakClassifiersLine.push_back(newWeakClass);
}

float StrongClassifier::applyStrongClassificator(std::vector<cv::Mat>* i, bool isPositive)
{
	int numOfElements = i->size();
	int numOfCorrectlyClassified = 0;

	std::vector<cv::Mat> imgs = *i;

	for (auto & classificator : weakClassifiersLine) {
		numOfCorrectlyClassified += applyFeature(classificator, &imgs, isPositive);		
	}

	return ((float)numOfElements-(float)numOfCorrectlyClassified) / (float)numOfElements;
}

