#include "GPUObjectDetectorPreprocess.h"

GPUObjectDetectorPreprocess::GPUObjectDetectorPreprocess(int windowSize, int windowStep, float pyrStep, bool measurePerformance) :
	slidingWindowSize(windowSize), slidingWindowStep(windowStep), pyramidStep(pyrStep), measurePerformance(measurePerformance)
{
	trainedClassifier = NULL;
	generator = std::default_random_engine(time(NULL));
	distribution = std::uniform_real_distribution<double>(0, 1);
}

GPUObjectDetectorPreprocess::~GPUObjectDetectorPreprocess()
{
}

void GPUObjectDetectorPreprocess::detectObjectInImage(std::string imgPath)
{
	
	// check if we should measure performance
	clock_t t;
	int numOfImages = 0;
	if (measurePerformance) {
		t = clock();
	}
	inputImagePath = imgPath;
	// go through all images in folder and prepare them for pyramide creation
	for (auto & p : std::experimental::filesystem::directory_iterator(inputImagePath)) {
		numOfImages++;
		cv::Mat curImg = cv::imread(p.path().string(), CV_LOAD_IMAGE_GRAYSCALE);
		curImg.convertTo(curImg, CV_64F);
		preprocessImage(&curImg);

		// print the data size which will be tranfered to GPU
		if (!measurePerformance) {
			std::cout << "Copy to GPU: " << std::endl;
			std::cout << "	ImageDataSize:            " << imagePyramid.size() * sizeof(double) << "B" << std::endl;
			std::cout << "	ImageDataDescriptorsSize: " << sizeof(int) * imagePyramidArrayDescriptorStep*4 << "B" << std::endl;
		}

		// we don't have real strong classifier, use random numbers
		if (trainedClassifier == NULL) {
			generateProbabilitiesForWeakClassifiers();
		}

		std::cout << "Calling GPU function." << std::endl;
		GPUObjectDetector(imagePyramidArray, imagePyramidArrayDescriptor, imagePyramidArrayDescriptorStep, imagePyramidArraySize, weakClassifierProbabilities, numOfWeakClassifiers);
		std::cout << "After GPU function." << std::endl;

		// GPU finished computing final image is in imagePyramidArray, show if now measuring performance	
		if (!measurePerformance) {
			std::vector<double> result;
			// get individual images
			for (int i = 0; i < imagePyramidArrayDescriptorStep; i++) {
				result.clear();
				int width = imagePyramidArrayDescriptor[i + imagePyramidArrayDescriptorStep];
				int height = 0;
				if (i + 1 >= imagePyramidArrayDescriptorStep)
					height = (imagePyramidArraySize - imagePyramidArrayDescriptor[i]) / width;
				else
					height = (imagePyramidArrayDescriptor[i + 1] - imagePyramidArrayDescriptor[i]) / width;
				// get data image to vector
				for (int j = imagePyramidArrayDescriptor[i]; j < imagePyramidArrayDescriptor[i + 1]; j++) {
					result.push_back(imagePyramidArray[j]);
				}
				// create image from that data
				cv::Mat resultImage = cv::Mat(height, width, CV_64F, result.data());
				resultImage.convertTo(resultImage, CV_8U);
				cv::namedWindow("Result");
				cv::imshow("Result", resultImage);
				cv::waitKey(0);
			}
		}
	}
	if (measurePerformance) {
		t = clock() - t;
		double miliSec = ((double)t) / (CLOCKS_PER_SEC / 1000); // in miliseconds
		std::cout << "-------------------------------" << std::endl;
		std::cout << "Number of images: " << numOfImages << ", time on CPU: " << miliSec << " ms" << std::endl;
	}
}

void GPUObjectDetectorPreprocess::preprocessImage(cv::Mat *inputImg)
{
	subSampleInputImage(inputImg);
	// build the pyramid on CPU
	buildPyramid(inputImg);
}

void GPUObjectDetectorPreprocess::subSampleInputImage(cv::Mat* i)
{
	cv::Size imgSize = i->size();
	if (imgSize.width > SUB_SAMPLED_INPUT_IMAGE_WIDTH  && SUB_SAMPLED_INPUT_IMAGE_WIDTH != -1) {
		float aspectRatio = float(imgSize.width) / float(imgSize.height);
		int newWidth = SUB_SAMPLED_INPUT_IMAGE_WIDTH;
		int newHeight = int((float)newWidth / aspectRatio);
		// we have new sizes, resize image, set new name and save
		cv::resize(*i, *i, cv::Size(newWidth, newHeight));
	}
}

void GPUObjectDetectorPreprocess::buildPyramid(cv::Mat* imagePyramidData)
{
	// variable contain the steps of pyramid
	cv::Mat currentImgData = *imagePyramidData;
	std::vector<std::pair<cv::Mat, cv::Rect>> pyramidStepsSizeAndImagesSize;
	// calculate the steps of pyramid
	cv::Size pyramidSize = cv::Size(0, 0);
	cv::Size newSize = cv::Size(0, 0);
	cv::Size currentSize = currentImgData.size();
	while (currentSize.width*this->pyramidStep > SLIDING_WINDOW_SIZE && currentSize.height > SLIDING_WINDOW_SIZE) {
		// if current size is not divisible by warp size, calculate size which will be; use current sze otherwise
		currentSize.width%WARP_SIZE != 0 ? newSize.width = int(ceil(currentSize.width / (float)WARP_SIZE) *WARP_SIZE) : newSize.width = currentSize.width;
		// the same for height
		currentSize.height%WARP_SIZE != 0 ? newSize.height = int(ceil(currentSize.height / (float)WARP_SIZE) *WARP_SIZE) : newSize.height = currentSize.height;
		// create rectangles that represent step of pyramid and image in it
		cv::Rect pyramidStep = cv::Rect(cv::Size(pyramidSize.width, 0), newSize);
		// add new width to pyramid size and edit height if needed
		pyramidSize.width += newSize.width;
		if (pyramidSize.height < newSize.height)
			pyramidSize.height = newSize.height;
		// save pyramid step size and image size in it
		pyramidStepsSizeAndImagesSize.push_back(std::make_pair(currentImgData, pyramidStep));

		// reduce the size of image
		resize(currentImgData, currentImgData, cv::Size(), this->pyramidStep, this->pyramidStep);
		// get size of next pyramid
		currentSize.width = currentImgData.size().width;
		currentSize.height = currentImgData.size().height;
	}
	
	// create pyramid with zeroes
	imagePyramidMat = cv::Mat::zeros(pyramidSize, CV_64F);
	// allocate the array for data and it's descriptor
	imagePyramidArrayDescriptor = new int[pyramidStepsSizeAndImagesSize.size()*4]; // create array for descriptors of data
	imagePyramidArrayDescriptorStep = pyramidStepsSizeAndImagesSize.size();

	// create pyramid matrix
	std::vector<std::pair<cv::Mat, cv::Rect>>::iterator pyramidStepsSizeAndImageSizeIterator = pyramidStepsSizeAndImagesSize.begin();
	int elemNumber = 0;
	imagePyramidArrayDescriptor[elemNumber] = 0;
	// go through all pyramid steps and create necessary data structures
	while (pyramidStepsSizeAndImageSizeIterator != pyramidStepsSizeAndImagesSize.end()) {
		cv::Mat stepImage = pyramidStepsSizeAndImageSizeIterator->first;
		cv::Rect stepSize = pyramidStepsSizeAndImageSizeIterator->second;
		cv::Mat pyrStep = cv::Mat::zeros(stepSize.size().height, stepSize.size().width, CV_64F);
		stepImage.copyTo(pyrStep(cv::Rect(cv::Point(0,0), stepImage.size())));
		if (!measurePerformance)// copy current step of pyramid to final cv::Mat
			pyrStep.copyTo(imagePyramidMat(stepSize));

		// add descriptors to descriptor array
		imagePyramidArrayDescriptor[elemNumber + imagePyramidArrayDescriptorStep] = stepSize.width;
		imagePyramidArrayDescriptor[elemNumber + (imagePyramidArrayDescriptorStep * 2)] = stepSize.width - stepImage.size().width;
		imagePyramidArrayDescriptor[elemNumber + (imagePyramidArrayDescriptorStep * 3)] = stepSize.width*stepImage.size().height;
		elemNumber++;
		if (elemNumber < imagePyramidArrayDescriptorStep)
			imagePyramidArrayDescriptor[elemNumber] = imagePyramidArrayDescriptor[elemNumber-1] + stepSize.width*stepSize.height; // set the beggining of another pyramid step

		// create 1D vector of all image Data
		if (pyrStep.isContinuous()) {
			std::vector<double> imagePyramidPart(pyrStep.begin<double>(),pyrStep.end<double>());
			imagePyramid.insert(imagePyramid.end(), imagePyramidPart.begin(), imagePyramidPart.end());
		}
		else {
			for (int i = 0; i < pyrStep.rows; ++i) {
				imagePyramid.insert(imagePyramid.end(), pyrStep.ptr<double>(i), pyrStep.ptr<double>(i) + pyrStep.cols);
			}
		}
		// get next pyramide step and different size image data
		++pyramidStepsSizeAndImageSizeIterator;
	}

	// get array of data from std::vector
	imagePyramidArray = imagePyramid.data();
	imagePyramidArraySize = imagePyramid.size();

	if (!measurePerformance) {
		cv::namedWindow("Pyramid");
		cv::Mat temp;
		imagePyramidMat.convertTo(temp, CV_8U);
		cv::imshow("Pyramid", temp);
		cv::waitKey(0);
	}
}

cv::Mat GPUObjectDetectorPreprocess::createMask(cv::Size pyrSize, cv::Rect imgPos)
{
	cv::Mat mask = cv::Mat::zeros(pyrSize, CV_8U);
	cv::rectangle(mask, imgPos, cv::Scalar(255), cv::FILLED);
	return mask;
}

void GPUObjectDetectorPreprocess::generateProbabilitiesForWeakClassifiers()
{
	numOfWeakClassifiers = STRONG_CLASSIFIER_SIZE;
	weakClassifierProbabilities = new double[numOfWeakClassifiers];
	// go through all weak classifier in strong one
	for (int i = 0; i < numOfWeakClassifiers; i++) {
		// get result from first weak classifier
		weakClassifierProbabilities[i] = distribution(generator);
	}
}
