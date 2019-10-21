#pragma once

// variables for learning
#define FALSE_POSITIVE_RATE 0.01f	// inicial false positive rate for first weak classifier	
#define FALSE_NEGATIVE_RATE 0.005f	// inicial false negative rate for forst weak classifier
#define IMAGE_WIDTH 24				// final size of input image for learning after sub-sampling
#define WINDOW_SIZE 24				// maximum size of Haarlike-feature kernel
#define LIMIT_WEAK_CLASSIFIERS 7	// maximum number of weak classifiers in Strong classifier
// paths to learning Sets
#define POSITIVE_SET_RAW_PATH "samples/positiveRaw"
#define NEGATIVE_SET_RAW_PATH "samples/negativeRaw"
#define POSITIVE_SET_PATH "samples\\positive\\"
#define NEGATIVE_SET_PATH "samples\\negative\\"


// variables for detection
#define SLIDING_WINDOW_STEP 4		// define how much we slide the detection window during detection
#define SLIDING_WINDOW_SIZE 24		// size of sliding window in which we are looking for object
#define PYRAMID_STEP 0.8f			// the with which we are sub-sampling the image -> instead of changing this size of sliding window, we are changing the image

#define SUB_SAMPLED_INPUT_IMAGE_WIDTH  480	// input image is sub-sampled to this width to speed up detection, do nothing if set to -1 or to bigger width than the actual width of input image

// path to set of images for detection
#define SET_FOR_DETECTION "detectionInput"

// variables for GPU detection preprocess
#define WARP_SIZE 32


// temporary variables, simulating strong classifier
#define REGION_DROP_CHANGE 0.9f				// init probability that the examined part of image is evaluated as "not contain object" 
#define REGION_DROP_RATE_SPEED_UP 1.0f		// function y=(1*REGION_DROP_RATE_SPEED_DOWN)/(x*REGION_DROP_RATE_SPEED_UP) is used for increase/decrease the REGION_DROP_CHANGE in each step(x) of strong classifier
#define REGION_DROP_RATE_SPEED_DOWN 1.0f	// reasonable values are in range <1,10>

#define STRONG_CLASSIFIER_SIZE 1000


#define MEASURE_PERFORMANCE false

