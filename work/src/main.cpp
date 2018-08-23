
// std
#include <iostream>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// project
#include "thin_plate.hpp"


using namespace cv;
using namespace std;



void testThinplate() {
	//Mat image = imread("work/res/checkerboard.jpg");
	Mat image = imread("work/res/lena.png");

	zhou::thinplate2d<float> tp;

	// corners
	tp.addPoint(Vec2f(1, 1), Vec2f(0, 0));
	tp.addPoint(Vec2f(510, 0), Vec2f(0, 0));
	tp.addPoint(Vec2f(0, 510), Vec2f(0, 0));
	tp.addPoint(Vec2f(510, 510), Vec2f(0, 0));

	// warp points
	tp.addPoint(Vec2f(200, 200), Vec2f(-20, -20));
	tp.addPoint(Vec2f(240, 240), Vec2f(20, 0));


	tp.computeWeights();


	Mat mapping(image.rows, image.cols, CV_32FC2);
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			mapping.at<Vec2f>(i, j) = Vec2f(j, i) - tp.evaluate(Vec2f(j, i));
		}
	}

	Mat warped;
	remap(image, warped, mapping, Mat(), INTER_LINEAR, BORDER_REPLICATE);


	// draw stuff
	for (int i = 0; i < tp.samples().size(); i++) {
		Point from(tp.samples()[i]);
		Point to(tp.samples()[i] + tp.values()[i]);
		Scalar c(255, 0, 0);
		circle(image, from, 3, c, 1); // , CV_AA
		line(image, from, to, c, 1);
		circle(warped, to, 3, c, 1);
		line(warped, from, to, c, 1);
	}

	imwrite("output/image.png", image);
	imwrite("output/warped.png", warped);
}








// main program
// 
int main( int argc, char** argv ) {




	// wait for a keystroke in the window before exiting
	waitKey(0);
}