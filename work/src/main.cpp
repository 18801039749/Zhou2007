
// std
#include <iostream>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// project
#include "thin_plate.hpp"
#include "ppa.hpp"
#include "featurepatch.hpp"


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



void testPPA() {

	Mat image = imread("work/res/mount_jackson.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat heightmap;
	image.convertTo(heightmap, CV_32FC1);


	ppa::FeatureGraph fg1(heightmap, 7, ppa::RIDGE_FEATURES);
	ppa::FeatureGraph fg2(heightmap, 7, ppa::VALLEY_FEATURES);
}




void testFeaturePatches() {
	Mat image = imread("work/res/mount_jackson.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat heightmap, cimage;
	image.convertTo(heightmap, CV_32FC1);
	cvtColor(image, cimage, COLOR_GRAY2BGR);

	ppa::FeatureGraph fg(heightmap, 7, ppa::RIDGE_FEATURES);

	// debug
	for (const auto &n : fg.nodes()) {
		circle(cimage, Point(n.second.p), 3, Scalar(0, 0, 225));
	}
	for (const auto &e : fg.edges()) {
		Point p = e.second.path[0];
		for (int i = 1; i < e.second.path.size(); i++) {
			Point next = e.second.path[i];
			line(cimage, p, next, Scalar(0, 255, 0));
			p = next;
		}
	}

	const int patch_size = 60;
	Point ps(patch_size, patch_size);

	auto patches = zhou::extractFeaturePatches(fg, patch_size);
	Mat heightmapborder;
	copyMakeBorder(cimage, heightmapborder, patch_size, patch_size, patch_size, patch_size, BORDER_CONSTANT);
	Mat heightmapcopy = heightmapborder.clone();
	Mat heightmappatch(patch_size, patch_size, heightmapborder.type());

	int count = 0;
	for (auto p : patches) {
		Rect roi(Point(p.center + Vec2f(patch_size / 2, patch_size / 2)), Size(patch_size, patch_size));
		heightmappatch = heightmapcopy(roi).clone();

		rectangle(heightmapborder, roi, Scalar(0, 255, 0));
		for (auto cp : p.controlpoints) {
			line(heightmapborder, Point(p.center) + ps, Point(cp) + ps, Scalar(255, 0, 0), 2);
			line(heightmappatch, Point(p.center) - roi.tl() + ps, Point(cp) - roi.tl() + ps, Scalar(255, 0, 0), 2);
		}

		ostringstream oss;
		oss << "output/patch" << count++ << ".png";
		imwrite(oss.str(), heightmappatch);
	}

	imwrite("output/patches.png", heightmapborder);
}



// main program
// 
int main( int argc, char** argv ) {

	//testThinplate();
	//testPPA();
	//testFeaturePatches();

	// wait for a keystroke in the window before exiting
	waitKey(0);
}