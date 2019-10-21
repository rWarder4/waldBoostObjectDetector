#include "HaarlikeFeatureGenerator.h"


HaarlikeFeatureGenerator::HaarlikeFeatureGenerator(int size)
{
	currentFeature = new wbs::HaarlikeFeature();
	imageSize = size;
}


HaarlikeFeatureGenerator::~HaarlikeFeatureGenerator()
{
}

wbs::HaarlikeFeature* HaarlikeFeatureGenerator::getNextFeature()
{
	createHorizontalEdgeFeature();
	return currentFeature;
}

void HaarlikeFeatureGenerator::resetHaarlikeGenerator()
{
	currentFeature->type = wbs::UNDEFINED_FEATURE;
}

wbs::HaarlikeFeature * HaarlikeFeatureGenerator::copyFeature(wbs::HaarlikeFeature *f)
{
	wbs::HaarlikeFeature* copyOfFeature = new wbs::HaarlikeFeature();
	
	copyOfFeature->type = f->type;
	copyOfFeature->xPos = f->xPos;
	copyOfFeature->yPos = f->yPos;
	copyFeatureAreaValues(copyOfFeature->a1, f->a1);
	copyFeatureAreaValues(copyOfFeature->a2, f->a2);
	copyFeatureAreaValues(copyOfFeature->a3, f->a3);
	copyFeatureAreaValues(copyOfFeature->a4, f->a4);

	return copyOfFeature;
}

void HaarlikeFeatureGenerator::copyFeatureAreaValues(wbs::HaarlikeFeatureAreas* to, wbs::HaarlikeFeatureAreas* from)
{
	to->x1 = from->x1;
	to->y1 = from->y1;
	to->x2 = from->x2;
	to->y2 = from->y2;
}



void HaarlikeFeatureGenerator::initCurrentFeaturePosition(int x, int y)
{
	currentFeature->xPos = x;
	currentFeature->yPos = y;
}

bool HaarlikeFeatureGenerator::createHorizontalEdgeFeature()
{
	// generating different type, do not continue
	if (currentFeature->type != wbs::HORIZONTAL_EDGE && currentFeature->type != wbs::UNDEFINED_FEATURE) {
		return false;
	}
	
	// generating next Haarlike feature of this type, find out what to generate now
	// is it first feature of this type? -> create first version
	if (currentFeature->xPos == -1 || currentFeature->yPos == -1) {
		currentFeature->type = wbs::HORIZONTAL_EDGE;
		currentFeature->xPos = 0;
		currentFeature->yPos = 0;
		
		currentFeature->a1->x1 = 0; // OO
		currentFeature->a1->y1 = 0; 
		currentFeature->a1->x2 = 1;
		currentFeature->a1->y2 = 0;

		currentFeature->a2->x1 = 0; // XX
		currentFeature->a2->y1 = 1;
		currentFeature->a2->x2 = 1;
		currentFeature->a2->y2 = 1;

		return true;
	}
	// is it possible to move current feature?
	if (currentFeature->xPos + currentFeature->a1->x2 + 1 < imageSize) { // stay on line, move to right
		currentFeature->xPos += 1;
		return true;
	}
	if (currentFeature->yPos + currentFeature->a2->y2 + 1 < imageSize) { // end of line, go to next line
		currentFeature->xPos = 0;
		currentFeature->yPos += 1;
		return true;
	}
	// move to first position, create next type feature
	currentFeature->xPos, currentFeature->yPos = 0;
	// impossible to expand, set next type and reset position
	if (currentFeature->a1->x2 + 2 >= imageSize && currentFeature->a2->y2 + 2 >= imageSize) {
		//std::cout << "change type" << std::endl;
		currentFeature->type = wbs::VERTICAL_EDGE;
		currentFeature->xPos = -1;
		currentFeature->yPos = -1;
		return false;
	}
	// expand in y, reset x if needed, set position to 0,0
	if (currentFeature->a1->x2 + 2 >= imageSize) {
		//std::cout << "expand in y, reset x:" << std::endl;
		currentFeature->a1->x2 = 1;
		currentFeature->a2->x2 = 1;
		
		currentFeature->a1->y2 += 1;
		currentFeature->a2->y1 += 1;
		currentFeature->a2->y2 += 2;

		initCurrentFeaturePosition();
		return true;
	}
	// expand in x
	//std::cout << "expand in x:" << std::endl;
	currentFeature->a1->x2 += 2;
	currentFeature->a2->x2 += 2;
	initCurrentFeaturePosition();
	return true;
}
