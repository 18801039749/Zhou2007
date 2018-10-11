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

		// construct matrix
		int row = 0;
		for (int i = 0; i < mask.rows; i++) {
			for (int j = 0; j < mask.cols; j++) {
				Point p(j, i);
				float pvalue = synthesis.at<float>(p);

				// ignore nan values
				if (isnan(pvalue)) continue;

				Point right_q = p + Point(1, 0);
				Point bottom_q = p + Point(0, 1);
				bool right = bound.contains(right_q) && !isnan(synthesis.at<float>(right_q));
				bool bottom = bound.contains(bottom_q) && !isnan(synthesis.at<float>(bottom_q));

				// if not masked, check if it is the boundary of masked pixels
				if (!mask.at<bool>(p)) {
					// for all neighbours
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

				// dx
				if (right) {
					// if p and q are a combination of no-mask and seam, use zero gradient
					if (
						(!mask.at<bool>(p) && seam_mask.at<bool>(right_q)) ||
						(seam_mask.at<bool>(p) && !mask.at<bool>(right_q))
					) {
						triplet_list.emplace_back(row, getid(right_q), 1);
						triplet_list.emplace_back(row, getid(p), -1);
						b_list.push_back(0);
						row++;
					}
					// else if one of them is masked, use a gradient
					else if (mask.at<bool>(p) || mask.at<bool>(right_q)) {
						triplet_list.emplace_back(row, getid(right_q), 1);
						triplet_list.emplace_back(row, getid(p), -1);
						b_list.push_back(synthesis.at<float>(right_q) - pvalue);
						row++;
					}
				}

				// dy
				if (bottom) {
					// if p and q are a combination of no-mask and seam, use zero gradient
					if (
						(!mask.at<bool>(p) && seam_mask.at<bool>(bottom_q)) ||
						(seam_mask.at<bool>(p) && !mask.at<bool>(bottom_q))
					) {
						triplet_list.emplace_back(row, getid(bottom_q), 1);
						triplet_list.emplace_back(row, getid(p), -1);
						b_list.push_back(0);
						row++;
					}
					// else if one of them is masked, use a gradient
					else if (mask.at<bool>(p) || mask.at<bool>(bottom_q)) {
						triplet_list.emplace_back(row, getid(bottom_q), 1);
						triplet_list.emplace_back(row, getid(p), -1);
						b_list.push_back(synthesis.at<float>(bottom_q) - pvalue);
						row++;
					}
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
	void placePatch(cv::Mat synthesis, cv::Mat patch, cv::Mat mask, cv::Vec2i pos) {
		using namespace std;
		using namespace cv;

		assert(patch.size() == mask.size());
		assert(synthesis.type() == CV_32FC1);
		assert(patch.type() == CV_32FC1);
		assert(mask.type() == CV_8UC1);

		// create a mask and seam_mask the same size as synthesis
		Mat synthesis_mask(synthesis.rows, synthesis.cols, CV_8UC1, Scalar(false));
		Point delta[4] = { {1,0}, {0,1}, {-1,0}, {0,-1} };
		Rect bound(Point(0, 0), synthesis.size());
		Mat seam_mask = synthesis_mask.clone();
		for (int i = 0; i < mask.rows; i++) {
			for (int j = 0; j < mask.cols; j++) {
				Point p(j + pos[0], i + pos[1]);
				// copy mask
				if (bound.contains(p) && !isnan(synthesis.at<float>(p)) && mask.at<bool>(i, j)) {
					synthesis_mask.at<bool>(p) = true;
				}
			}
		}

		// create seam_mask
		for (int i = 0; i < mask.rows; i++) {
			for (int j = 0; j < mask.cols; j++) {
				Point p(j + pos[0], i + pos[1]);
				if (bound.contains(p) && synthesis_mask.at<bool>(p)) {
					// check if there's a seam
					for (int d = 0; d < 4; d++) {
						Point q = p + delta[d];
						// if the neighbour 'q' is not masked, and is not nan, then 'p' is a seam
						if (bound.contains(q) && !isnan(synthesis.at<float>(q)) && !synthesis_mask.at<bool>(q)) {
							seam_mask.at<bool>(q) = true;
						}
					}
				}
			}
		}

		// place value into synthesis
		for (int i = 0; i < mask.rows; i++) {
			for (int j = 0; j < mask.cols; j++) {
				Point p(j + pos[0], i + pos[1]);
				if (bound.contains(p) && mask.at<bool>(i, j)) {
					synthesis.at<float>(p) = patch.at<float>(i, j);
				}
			}
		}

		// remove the seam
		poissonSeamRemoval(synthesis, synthesis_mask, seam_mask);

		// done
	}
}