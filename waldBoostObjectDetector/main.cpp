#include <iostream>

#include "CPUObjectDetector.h"
#include "HaarlikeFeatureGenerator.h"
#include "StrongClassifier.h"
#include "WBSSettings.h"
#include "GPUObjectDetectorPreprocess.h"


int main() {

	CPUObjectDetector* cpuDetector = new CPUObjectDetector(SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STEP, PYRAMID_STEP, MEASURE_PERFORMANCE);
	cpuDetector->detectObjectInImage(SET_FOR_DETECTION);

	GPUObjectDetectorPreprocess* gpuDetector = new GPUObjectDetectorPreprocess(SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STEP, PYRAMID_STEP, MEASURE_PERFORMANCE);
	gpuDetector->detectObjectInImage(SET_FOR_DETECTION);

	return 0;
}