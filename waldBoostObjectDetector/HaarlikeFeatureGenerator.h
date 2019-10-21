#pragma once

#include <opencv2\core\core.hpp>
#include <iostream>

#include "WBSStructs.h"


class HaarlikeFeatureGenerator
{

private:
	wbs::HaarlikeFeature* currentFeature;
	int imageSize;

	void initCurrentFeaturePosition(int x = 0, int y = 0);

	bool createHorizontalEdgeFeature();
	void createVerticalEdgeFeature();
	void createHorizontalLineFeature();
	void createVerticalLineFeature();
	void createDiagonalTopLeftFeature();
	void createDiagonalBottomLeftFeature();

public:
	HaarlikeFeatureGenerator(int size = 24);
	~HaarlikeFeatureGenerator();

	wbs::HaarlikeFeature* getNextFeature();
	void resetHaarlikeGenerator();

	wbs::HaarlikeFeature* copyFeature(wbs::HaarlikeFeature *f);
	void copyFeatureAreaValues(wbs::HaarlikeFeatureAreas* from, wbs::HaarlikeFeatureAreas* to);
};

