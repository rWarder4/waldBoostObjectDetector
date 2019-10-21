#include "CPUObjectDetector.h"



CPUObjectDetector::CPUObjectDetector(int windowSize, int windowStep, float pyrStep, bool measurePerformance) :
	slidingWindowSize(windowSize), slidingWindowStep(windowStep), pyramidStep(pyrStep), slidingWindowPosition(std::make_pair(0, 0)), measurePerformance(measurePerformance)
{
	generator = std::default_random_engine(time(NULL));
	distribution = std::uniform_real_distribution<double>(0, 1);
}


CPUObjectDetector::~CPUObjectDetector()
{
}

void CPUObjectDetector::detectObjectInImage(std::string imgPath)
{
	// check if we should measure performance
	clock_t t;
	int numOfImages = 0;
	if (measurePerformance) {
		t = clock();
	}
	// go through all images in folder and locate the objects with rectangle
	for (auto & p : std::experimental::filesystem::directory_iterator(imgPath)) {
		numOfImages++;
		setNextImage(&cv::imread(p.path().string(), CV_LOAD_IMAGE_GRAYSCALE));
		// for loaded image create Pyramid
		preprocessImage();

		/*std::cout << imagePyramid.size() << std::endl;
		cv::namedWindow("PyramidStep");
		for (auto &p : imagePyramid) {
			cv::imshow("PyramidStep", p);
			cv::waitKey(100);
		}*/
			
		// apply sliding window on all images from pyramid
		getNextImageFromPyramid();
		initSlidingWindowPosition();
		int temp = 0;
		while (!processedImg.empty()) {
			// check all available sliding window positions
			while (slidingWindowPosition.first != -1 && slidingWindowPosition.second != -1) {
				// use waldboost on processed image under sliding window
				applyStrongClassifier();
				// move window
				setNextSlidingWindowPosition();
				temp++;
			}
			// get next image and init
			getNextImageFromPyramid();
			initSlidingWindowPosition();
		}
		
		std::cout << temp << std::endl;

		if (!measurePerformance && foundedObjects.size() != 0) // check the founded object
			showImageWithFoundedObjects();

		initForNextInputImage();
	}
	if (measurePerformance) {
		t = clock() - t;
		double miliSec = ((double)t) / (CLOCKS_PER_SEC/1000); // in miliseconds
		std::cout << "-------------------------------" << std::endl;
		std::cout << "Number of images: "<< numOfImages << ", time on CPU: " << miliSec << " ms" << std::endl;

	}
}

void CPUObjectDetector::setStrongClassifier(StrongClassifier * s)
{
	this->trainedClassifier = s;
}

void CPUObjectDetector::initSlidingWindowPosition()
{
	slidingWindowPosition.first = 0;
	slidingWindowPosition.second = 0;
}

void CPUObjectDetector::setNextSlidingWindowPosition()
{
	// check if we can move horizontally, otherwise check if we can move vertically, finally return -1,-1 as position to finish process the image
	int horizontalEndOfWindow = slidingWindowPosition.first + slidingWindowSize + slidingWindowStep;
	int verticalEndOfWindow = slidingWindowPosition.second + slidingWindowSize + slidingWindowStep;
	if (horizontalEndOfWindow <= processedImgSize.width) { // move horizontally
		slidingWindowPosition.first += slidingWindowStep;
	}
	else if (verticalEndOfWindow <= processedImgSize.height) { // move vertically
		slidingWindowPosition.first = 0;
		slidingWindowPosition.second += slidingWindowStep;
	}
	else { // image processed
		slidingWindowPosition.first = -1;
		slidingWindowPosition.second = -1;
	}
}

void CPUObjectDetector::getNextImageFromPyramid()
{
	if (pyramidIterator == imagePyramid.end()) {
		processedImg = cv::Mat();
	}
	else {
		processedImg = *pyramidIterator;
		// update iterator
		++pyramidIterator;
	}
	// calculate size
	this->processedImgSize = processedImg.size();	
}

void CPUObjectDetector::setNextImage(cv::Mat* i)
{
	this->inputImg = *i;
	// sub-sample if needed
	subSampleInputImage();
}

void CPUObjectDetector::preprocessImage()
{
	// create pyramides
	cv::Mat pyramidStep = this->inputImg;
	cv::Size imgSize = pyramidStep.size();
	while (imgSize.width > this->slidingWindowSize) {
		// add image to pyramid
		imagePyramid.push_back(pyramidStep);
		// reduce the size of image
		resize(pyramidStep, pyramidStep, cv::Size(), this->pyramidStep, this->pyramidStep);
		imgSize = pyramidStep.size();
	}
	// set iterator to begining of pyramid array
	this->pyramidIterator = this->imagePyramid.begin();
}

void CPUObjectDetector::subSampleInputImage()
{
	// sub-sample and keep aspect ratio
	cv::Size inputImageSize = this->inputImg.size();
	if (inputImageSize.width > SUB_SAMPLED_INPUT_IMAGE_WIDTH  && SUB_SAMPLED_INPUT_IMAGE_WIDTH != -1) {
		float aspectRatio = float(inputImageSize.width) / float(inputImageSize.height);
		int newWidth = SUB_SAMPLED_INPUT_IMAGE_WIDTH;
		int newHeight = float(newWidth) / aspectRatio;
		// we have new sizes, resize image, set new name and save
		cv::resize(this->inputImg, this->inputImg, cv::Size(newWidth, newHeight));
	}
}

void CPUObjectDetector::initForNextInputImage()
{
	imagePyramid.clear();
	foundedObjects.clear();
	slidingWindowPosition = std::make_pair(0, 0);
}

void CPUObjectDetector::applyStrongClassifier()
{
	// create roi
	cv::Rect roi = cv::Rect(this->slidingWindowPosition.first, this->slidingWindowPosition.second, this->slidingWindowSize, this->slidingWindowSize);
	// get image under current window
	cv::Mat prImg(processedImg, roi);
	/*cv::namedWindow("Result");
	cv::imshow("Result", prImg);*/
	// calculate integral image
	cv::Mat integral;
	cv::integral(prImg, integral);
	/*cv::imshow("Result2", integral);
	cv::waitKey(0);*/
	// go through all weak classifier in strong one
	for (int i = 1; i < STRONG_CLASSIFIER_SIZE+1; i++) {
		float dropThreshold = REGION_DROP_CHANGE * ((1 * REGION_DROP_RATE_SPEED_DOWN) / (i*REGION_DROP_RATE_SPEED_UP));
		// get result from first weak classifier
		float result = float(distribution(generator));
		if (result < dropThreshold) {
			return;
		}
	}
	// there is high probability that this region is founded object
	addFoundedObject();
}

void CPUObjectDetector::addFoundedObject()
{
	this->foundedObjects.push_back(std::make_pair(
		cv::Rect(this->slidingWindowPosition.first,
			this->slidingWindowPosition.second,
			slidingWindowSize, slidingWindowSize),
		processedImg.size()));
}

void CPUObjectDetector::showImageWithFoundedObjects()
{
	cv::namedWindow("Result");
	for (auto &p : foundedObjects) {
		cv::Rect r = p.first;
		cv::Size imgSizeSearch = p.second;
		cv::Size imgRealSize = inputImg.size();
		// recalculate real size
		float widthRatio = imgRealSize.width / imgSizeSearch.width;
		float heightRatio = imgRealSize.height / imgSizeSearch.height;
		r.width *= widthRatio;
		r.height *= heightRatio;

		cv::rectangle(inputImg, r, cv::Scalar(255, 0, 0), 2);
	}
	cv::imshow("Result", inputImg);
	cv::waitKey(0);
}
