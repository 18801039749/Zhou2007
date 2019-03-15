#pragma once


#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

// opencv
#include <opencv2/core.hpp>

// project
#include "eigen.hpp"


namespace zhou {

	// Given the heightmap, mask and mask offset
	// modify the heightmap to seamlessly fit in with surroundings
	// TODO reform to only use values inside the mask?
	void poissonSeamRemoval(cv::Mat synthesis, cv::Mat mask, cv::Mat seam_mask) {
		using namespace cv;
		using namespace std;

		assert(synthesis.size() == mask.size());
		assert(synthesis.size() == seam_mask.size());
		assert(synthesis.type() == CV_32FC1);
		assert(mask.type() == CV_8UC1);
		assert(seam_mask.type() == CV_8UC1);

		
		// directions and bound
		Point delta[4] = { {1,0}, {0,1}, {-1,0}, {0,-1} };
		Rect bound(Point(0,0), mask.size());

		// indexing
		Mat pointToid(mask.rows, mask.cols, CV_32SC1, Scalar(-1));
		vector<Point> idToPoint;

		// helper function to create an id
		auto getid = [&](const Point &p) {
			int id = pointToid.at<int>(p);
			if (id < 0) { // create id
				id = idToPoint.size();
				idToPoint.push_back(p);
				pointToid.at<int>(p) = id;
			}
			return id;
		};


		// Prepare for constructing sparse matrix
		// Ax = b, where the A-col, or b-row is the point in matrix
		vector<Eigen::Triplet<float>> triplet_list; // triplet (row, col, val) in A
		vector<float> b_list; // value in b
		triplet_list.reserve(1024); // reserve some amount of space
		b_list.reserve(1024);


		//cout << "Initializing" << endl;


		// note to self:
		// triplet_list.emplace_back(ROW, COL_ID, VALUE);

		// construct matrix
		int row = 0;
		for (int i = 0; i < mask.rows; i++) {
			for (int j = 0; j < mask.cols; j++) {
				Point p(j, i);
				float pvalue = synthesis.at<float>(p);

				// ignore nan values
				if (isnan(pvalue)) continue;


				// if not masked, but one of the neighbours is masked
				// then this is a dirichlet boundry condition
				if (!mask.at<bool>(p)) {
					for (int d = 0; d < 4; d++) {
						Point q = p + delta[d];
						if (bound.contains(q) && mask.at<bool>(q)) {
							// put in sparse matrix as boundary condition
							triplet_list.emplace_back(row, getid(p), 1);
							b_list.push_back(pvalue);
							row++;
							break;
						}
					}
				}

				// calculate dx
				// either p or left is a masked value
				Point left = p + Point(-1, 0);
				if (bound.contains(left) && !isnan(synthesis.at<float>(left)) && (mask.at<bool>(p) || mask.at<bool>(left))) {
					triplet_list.emplace_back(row, getid(p), 1);
					triplet_list.emplace_back(row, getid(left), -1);
					if (seam_mask.at<bool>(p) && seam_mask.at<bool>(left)) {
						b_list.push_back(0);
					} else {
						b_list.push_back(pvalue - synthesis.at<float>(left));
					}
					row++;
				}


				// calculate dy
				// either p or top is a masked value
				Point top = p + Point(0, -1);
				if (bound.contains(top) && !isnan(synthesis.at<float>(top)) && (mask.at<bool>(p) || mask.at<bool>(top))) {
					triplet_list.emplace_back(row, getid(p), 1);
					triplet_list.emplace_back(row, getid(top), -1);
					if (seam_mask.at<bool>(p) && seam_mask.at<bool>(top)) {
						b_list.push_back(0);
					}
					else {
						b_list.push_back(pvalue - synthesis.at<float>(top));
					}
					row++;
				}

			}
		}


		//cout << "Building" << endl;

		// Build the sparse matrix (nxn)
		Eigen::SparseMatrix<float> A(row, idToPoint.size());
		Eigen::VectorXf x(idToPoint.size());
		Eigen::VectorXf b = Eigen::Map<Eigen::VectorXf>(b_list.data(), b_list.size());

		A.setFromTriplets(triplet_list.begin(), triplet_list.end());

		//cout << "Compressing" << endl;

		A.makeCompressed();


		// Solve
		// SparseQR<SparseMatrix<float>, COLAMDOrdering<int>> solver;
		// ConjugateGradient<SparseMatrix<float>> solver;
		Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<float>> solver;

		// cout << "Analyzing" << endl;
		// solver.analyzePattern(A);

		// cout << "Factorizing" << endl;
		// solver.factorize(A);

		// cout << "Computing" << endl;
		solver.compute(A);

		//cout << "Solving " << endl;
		x = solver.solve(b);
		//cout << "Finished" << endl;


		// apply the results to the synthesis
		for (int i = 0; i < idToPoint.size(); ++i) {
			Point p = idToPoint.at(i);
			//if (mask.at<bool>(p)) {
				synthesis.at<float>(p) = x(i);
			//}
		}
	}




	// place patch using the graphcut mask provided
	// uses seam removal
	// assumes patch is non null
	void placePatch(cv::Mat synthesis, cv::Mat patch, cv::Mat mask, cv::Vec2i pos) {
		using namespace std;
		using namespace cv;

		assert(patch.size() == mask.size());
		assert(synthesis.type() == CV_32FC1);
		assert(patch.type() == CV_32FC1);
		assert(mask.type() == CV_8UC1);

		Point delta[4] = { {1,0}, {0,1}, {-1,0}, {0,-1} };
		Rect patchBound(Point(0, 0), patch.size());
		Rect synthesisBound(Point(0, 0), synthesis.size());

		// create synthesis-sized overlap mask and seam mask
		Mat synthesis_overlap(synthesis.rows, synthesis.cols, CV_8UC1, Scalar(false));
		Mat seam_mask(synthesis.rows, synthesis.cols, CV_8UC1, Scalar(false));
		for (int i = 0; i < mask.rows; i++) {
			for (int j = 0; j < mask.cols; j++) {
				Point p(j + pos[0], i + pos[1]);
				if (synthesisBound.contains(p)) {
					// entire overlap area
					if (!isnan(synthesis.at<float>(p))) {
						synthesis_overlap.at<bool>(p) = true;
					}

					// mask value placement
					if (mask.at<bool>(i, j)) {
						synthesis.at<float>(p) = patch.at<float>(i, j);
					}

					// a seam is pixel in the overlap area and on the cut boundry
					for (int d = 0; d < 4; d++) {
						Point neighbour = Point(j, i) + delta[d];
						if (patchBound.contains(neighbour) && mask.at<bool>(i, j) != mask.at<bool>(neighbour)) {
							seam_mask.at<bool>(p) = true;
						}
					}
				}
			}
		}

		// remove the seam
		poissonSeamRemoval(synthesis, synthesis_overlap, seam_mask);


		//// debug
		//Mat maskImage(synthesis.rows, synthesis.cols, CV_8UC3, Scalar(0));
		//for (int i = 0; i < maskImage.rows; i++) {
		//	for (int j = 0; j < maskImage.cols; j++) {
		//		if (synthesis_overlap.at<bool>(i, j))
		//			maskImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
		//		if (seam_mask.at<bool>(i, j))
		//			maskImage.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
		//	}
		//}
		//imwrite("output/maskiamge.png", maskImage);


		// done
	}
}