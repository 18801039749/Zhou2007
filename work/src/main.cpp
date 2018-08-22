
// std
#include <iostream>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// project
#include "thin_plate.hpp"


using namespace cv;
using namespace std;


// main program
// 
int main( int argc, char** argv ) {


	zhou::thinplate2d<float> tp;
	tp.addPoint(Vec2f(1, 2), Vec2f(0, 0));
	tp.addPoint(Vec2f(4, 3), Vec2f(0, 0));
	tp.addPoint(Vec2f(2, 4), Vec2f(0, 0));
	tp.addPoint(Vec2f(3, 5), Vec2f(0, 0));
	tp.computeWeights();


		
	Mat m;

	Mat mapping(m.rows, m.cols, CV_32FC2);
	for (int i = 0; i < m.rows; ++i) {
		for (int j = 0; j < m.cols; ++j) {
			mapping.at<Vec2f>(i, j) = tp.evaluate(Vec2f(j, i));
		}
	}

	// wait for a keystroke in the window before exiting
	waitKey(0);
}