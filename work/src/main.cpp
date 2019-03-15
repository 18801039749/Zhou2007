
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
#include "terrain.hpp"
#include "graphcut.hpp"
#include "patchmerge.hpp"
#include "zhou.hpp"


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


	ppa::FeatureGraph fg1(heightmap, 5, 7, ppa::RIDGE_FEATURES);
	//ppa::FeatureGraph fg2(heightmap, 20, 7, ppa::VALLEY_FEATURES);

}




void testFeaturePatches() {
	Mat image = imread("work/res/mount_jackson.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat heightmap, cimage;
	image.convertTo(heightmap, CV_32FC1);
	cvtColor(image, cimage, COLOR_GRAY2BGR);

	ppa::FeatureGraph fg(heightmap, 10, 7, ppa::RIDGE_FEATURES);

	// debug
	Mat fullimage = cimage.clone();
	for (const auto &n : fg.nodes()) {
		circle(fullimage, Point(n.second.p), 3, Scalar(0, 0, 225));
	}
	for (const auto &e : fg.edges()) {
		Point p = e.second.path[0];
		for (int i = 1; i < e.second.path.size(); i++) {
			Point next = e.second.path[i];
			line(fullimage, p, next, Scalar(0, 255, 0), 2);
			p = next;
		}
	}
	imwrite("output/patches.png", fullimage);


	int count = 0;
	const int patch_size = 60;
	auto patches = zhou::extractFeaturePatches(fg, patch_size);
	for (auto fp : patches) {
		ostringstream oss;
		oss << "output/patch" << count++ << ".png";
		imwrite(oss.str(), zhou::fpatch2img(fp, patch_size, cimage));
	}
}



void testGraphCut() {
	Mat image = imread("work/res/brick.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat fimage;
	image.convertTo(fimage, CV_32FC1);

	fimage(Range(0, image.rows), Range(image.cols / 2, image.cols)).setTo(numeric_limits<float>::quiet_NaN());
	fimage(Range(image.rows / 2, image.rows), Range(0, image.cols)).setTo(numeric_limits<float>::quiet_NaN());
	Mat patch = fimage(Range(0, 64), Range(0, 64)).clone();

	Vec2i pos(image.cols / 2 - patch.cols / 2, image.rows / 2 - patch.rows / 2);
	float cost;
	Mat cut = zhou::graphcut(fimage, patch, pos, &cost);
	Mat cutpatch = patch.clone();

	for (int i = 0; i < patch.rows; i++) {
		for (int j = 0; j < patch.cols; j++) {
			if (cut.at<bool>(i, j)) {
				fimage.at<float>(i + pos[1], j + pos[0]) = patch.at<float>(i, j);
			}
			else {
				cutpatch.at<float>(i, j) = 0;
			}
		}

	}

	cout << "cost : " << cost << endl;
	imwrite("output/synthesis.png", fimage);
	imwrite("output/patch.png", patch);
	imwrite("output/cut.png", cutpatch);
}


void testSeamRemoval() {
	Mat image = imread("work/res/brick.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat fimage;
	image.convertTo(fimage, CV_32FC1);

	fimage(Range(0, image.rows), Range(image.cols / 2, image.cols)).setTo(numeric_limits<float>::quiet_NaN());
	Mat patch = fimage(Range(0, 128), Range(0, 128)).clone();

	Vec2i pos(image.cols / 2 - patch.cols / 2, image.rows / 2 - patch.rows / 2);
	Mat cut = zhou::graphcut(fimage, patch, pos);

	zhou::placePatch(fimage, patch, cut, pos);


	imwrite("output/synthesis.png", fimage);
	imwrite("output/patch.png", patch);


	//Mat image(2, 5, CV_32FC1, 1);
	//Mat mask(2, 5, CV_8UC1, Scalar(false));
	//Mat seam(2, 5, CV_8UC1, Scalar(false));
	//for (int i = 0; i < image.rows; i++) {
	//	for (int j = 0; j < image.cols; j++) {
	//		image.at<float>(i, j) = j*10 + 10;
	//		if (j >= 1 && j < 4) {
	//			if (j == 1) {
	//				seam.at<bool>(i, j) = true;
	//			}
	//			mask.at<bool>(i, j) = true;
	//		}
	//	}
	//}

	//cout << "before" << image << endl;
	//zhou::poissonSeamRemoval(image, mask, seam);
	//cout << "after" << image << endl;
}



void testSynthesis() {
	
	//zhou::terrain test_terrain = zhou::terrainReadImage("work/res/mount_jackson.png", 0, 255, 1);
	//zhou::terrain test_terrain = zhou::terrainReadTIFF("work/res/mt_fuji_n035e138.tif");
	//zhou::terrain test_terrain = zhou::terrainReadTIFF("work/res/mount_jackson_n39_w107_3arc.tif");
	zhou::terrain test_terrain = zhou::terrainReadTIFF("work/res/southern_alps_s045e169.tif");

	Mat sketchmap, image2 = imread("work/res/fractal_terrain.png", CV_LOAD_IMAGE_GRAYSCALE);
	image2.convertTo(sketchmap, CV_32FC1);

	zhou::synthesisparams p;
	p.ppaGridSpacing = 30;
	Mat synthesis = zhou::synthesize(test_terrain.heightmap, sketchmap, p);

	imwrite("output/salps_synth.png", zhou::heightmapToImage(synthesis));
	zhou::terrainWriteTxt("output/salps_synth.asc", zhou::terrain(synthesis, test_terrain.spacing));
}


void testRotation() {

	Vec2f point(11, 10);
	Vec2f center(10, 10);
	Vec2f outpath1 = center - point;
	Vec2f outpath2(0, 1);
	float sign = copysign(1, outpath2[0] * outpath1[1] - outpath2[1] * outpath1[0]);
	float angle = sign * acos(outpath1.dot(outpath2));

	float c = cos(angle);
	float s = sin(angle);


	Vec2f p = point - center;
		
	p = Vec2f(c * p[0] - s * p[1], s * p[0] + c * p[1]);
	p += center;

	cout << "p" << p << endl; // expect (10, 11)
}



// main program
// 
int main( int argc, char** argv ) {

	//testThinplate();
	//testPPA();
	//testFeaturePatches();
	//testRotation();
	//testGraphCut();
	//testSeamRemoval();
	testSynthesis();

	// wait for a keystroke in the window before exiting
	waitKey(0);
}