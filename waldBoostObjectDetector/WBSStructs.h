#pragma once


namespace wbs {
	enum HaarlikeFeatureType {
		UNDEFINED_FEATURE = -1,
		HORIZONTAL_EDGE = 0,
		VERTICAL_EDGE,
		HORIZONTAL_LINE,
		VERTICAL_LINE,
		DIAGONAL_TOP_LEFT,
		DIAGONAL_BOTTOM_LEFT,
	};

	struct HaarlikeFeatureAreas {
		int x1;
		int y1;
		int x2;
		int y2;
		HaarlikeFeatureAreas(int a = -1, int b = -1, int c = -1, int d = -1) : x1(a), y1(b), x2(c), y2(d) {}
	};

	struct HaarlikeFeature {
		HaarlikeFeatureType type = UNDEFINED_FEATURE;
		int xPos = -1;
		int yPos = -1;
		HaarlikeFeatureAreas* a1;
		HaarlikeFeatureAreas* a2;
		HaarlikeFeatureAreas* a3; // used if line or diagonal type is chosen
		HaarlikeFeatureAreas* a4; // used if diagonal type is chosen
		HaarlikeFeature() {
			a1 = new HaarlikeFeatureAreas();
			a2 = new HaarlikeFeatureAreas();
			a3 = new HaarlikeFeatureAreas();
			a4 = new HaarlikeFeatureAreas();
		}
	};
}