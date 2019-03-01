#pragma once

// std
#include <unordered_set>
#include <queue>
#include <vector>

// opencv
#include <opencv2/core.hpp>

// project
#include "ppa.hpp"


namespace zhou {

	struct fpatch {
		cv::Vec2f center; // center relative to the original data
		std::vector<cv::Vec2f> controlpoints; // outgoing points relative to the center (but does not contain the center)
	};


	inline bool circleLineIntersection(cv::Vec2f center, float radius, cv::Vec2f start, cv::Vec2f end, cv::Vec2f &out) {
		cv::Vec2f d = end - start; // direction
		cv::Vec2f f = start - center; // distance to center

		float a = d.dot(d);
		float b = 2 * f.dot(d);
		float c = f.dot(f) - radius * radius;
		float discrim = b * b - 4 * a * c;

		if (discrim >= 0) {
			discrim = sqrt(discrim);
			float t0 = (-b - discrim) / (2 * a);
			float t1 = (-b + discrim) / (2 * a);
			if (t0 >= 0 && t0 <= 1) {
				out = start + t0 * d;
				return true;
			}
			if (t1 >= 0 && t1 <= 1) {
				out = start + t1 * d;
				return true;
			}
		}

		return false;
	}


	// helper method that returns the first point a path crosses the circle
	// returns false if there is none
	inline bool circlePathIntersection(cv::Vec2f center, float radius, std::vector<cv::Vec2f> path, bool reversePath, bool extend, cv::Vec2f &out_intersection) {
		if (reversePath) reverse(path.begin(), path.end());

		assert(!path.empty());
		assert(radius > 0);
		assert(norm(center, path[0]) < radius);

		using namespace cv;
		using namespace std;

		for (int i = 0; i < path.size() - 1; ++i) {
			if (circleLineIntersection(center, radius, path[i], path[i + 1], out_intersection)) {
				return true;
			}
		}

		if (extend) {

			// no intersection found, so extend the path from the center to the last point and beyond
			Vec2f start = path[path.size() - 1];
			Vec2f d = start - center;
			double n = norm(d);
			if (n > 0) d *= 2 * radius / n;
			else return false;
			Vec2f end = start + d;

			if (!circleLineIntersection(center, radius, start, end, out_intersection))
				cerr << "ERROR in intersection code" << endl;

			return true;
		}

		return false;
	}



	// process the node "current" that came from "parent", trying to return the point any edges leave the "center radius"
	inline std::vector<cv::Vec2f> proccessNode(const ppa::FeatureGraph &features, cv::Vec2f center, float radius, int parent, int current) {

		assert(norm(center, features.nodes().at(current).p) < radius);

		using namespace cv;
		using namespace std;


		vector<Vec2f> intersections;

		for (int edgeid : features.nodes().at(current).edges) {
			// dont processes edge that contains parent
			const ppa::FeatureEdge &edge = features.edges().at(edgeid);
			if (edge.other(current) == parent) continue;

			// check outgoing edge for intersection
			Vec2f intersection;
			if (circlePathIntersection(center, radius, edge.path, (edge.node_start != current), false, intersection)) {
				intersections.push_back(intersection - center);
			}

			// otherwise recursively serch for intersections along the graph
			else {
				vector<Vec2f> p = proccessNode(features, center, radius, current, edge.other(current));
				intersections.insert(intersections.end(), p.begin(), p.end());
			}
		}

		return intersections;
	}


	/*
	It is debateable what Zhou originally ment in his paper about how this should be implemented. 
	*/
	inline std::vector<fpatch> extractFeaturePatches(const ppa::FeatureGraph &features, int patch_size) {
		assert(patch_size > 1);

		using namespace cv;
		using namespace std;

		int radius = patch_size / 2;
		vector<fpatch> featurepatches;



		// breadth-first
		unordered_set<int> visited;
		queue<int> toprocess;
		for (const auto &nodeitem : features.nodes()) {
			int id = nodeitem.first;
			if (visited.find(id) != visited.end()) continue;
			toprocess.push(id);

			while (!toprocess.empty()) {
				int nodeid = toprocess.front();
				Vec2f p = features.nodes().at(nodeid).p;
				toprocess.pop();
				if (visited.find(nodeid) != visited.end()) continue;
				visited.insert(nodeid);


				// end-features and branch features
				//
				std::vector<Vec2f> controlpoints = proccessNode(features, p, radius, -1, nodeid);
				if (!controlpoints.empty()) {
					featurepatches.push_back(fpatch{ p, controlpoints });
				}


				for (int edgeid : features.nodes().at(nodeid).edges) {
					const auto &edge = features.edges().at(edgeid);
					int othernodeid = edge.other(nodeid);
					if (visited.find(othernodeid) != visited.end()) continue;
					toprocess.push(othernodeid);

					// path-features
					//
					vector<Vec2f> previousPath; // reverse order
					vector<Vec2f> nextPath = edge.path;
					// traverse edge with the path before and patch after
					previousPath.push_back(nextPath.back());
					nextPath.pop_back();
					float distance = 0;

					// while there is a line segment to traverse
					while (nextPath.size() >= 2) {
						Vec2f center = nextPath.back();
						distance += norm(previousPath.back(), center);
						previousPath.push_back(center);

						// progress in steps of size "radius"
						if (distance > radius) {
							distance -= radius;

							// create patch here
							vector<Vec2f> controlpoints;
							Vec2f point;
							circlePathIntersection(center, radius, previousPath, true, true, point);
							controlpoints.push_back(point - center);
							circlePathIntersection(center, radius, nextPath, true, true, point);
							controlpoints.push_back(point - center);
							featurepatches.push_back(fpatch{ center, controlpoints });
						}
						nextPath.pop_back();
					}
					// 
					// path-features
				}

			}
		}

		return featurepatches;
	}

	
	inline std::vector<cv::Mat> extractNonfeaturePatches(cv::Mat examplemap, std::vector<fpatch> featurePatches, int patch_size) {
		assert(patch_size > 1);
		assert(!examplemap.empty());
		assert(examplemap.type() == CV_32FC1);

		using namespace cv;
		using namespace std;

		int hp1 = patch_size / 2;
		int hp2 = patch_size - hp1;

		Mat mask(examplemap.rows, examplemap.cols, CV_8UC1, Scalar(false));
		Rect bound(Point(0, 0), mask.size());
		for (fpatch fp : featurePatches) {
			for (int i = fp.center[1] - hp1; i < fp.center[1] + hp2; ++i) {
				for (int j = fp.center[0] - hp1; j < fp.center[0] + hp2; ++j) {
					if (bound.contains(Point(j, i))) {
						mask.at<bool>(i, j) = true;
					}
				}
			}
		}

		vector<Mat> nonfeaturePatches;
		for (int i = 0; i < mask.rows - patch_size; i += patch_size / 2) {
			for (int j = 0; j < mask.cols - patch_size; j += patch_size / 2) {
				if (!mask.at<bool>(i + hp1, j + hp1)) {
					nonfeaturePatches.push_back(examplemap(Range(i, i + patch_size), Range(j, j + patch_size)));
				}
			}
		}

		return nonfeaturePatches;
	}


	// source must be a BGR 8UC3
	inline cv::Mat fpatch2img(const fpatch &fp, int patch_size, cv::Mat source) {
		using namespace cv;

		assert(!source.empty());
		assert(source.type() == CV_8UC3);
		
		Rect roi(Point(fp.center - Vec2f(patch_size / 2, patch_size / 2)), Size(patch_size, patch_size));

		Mat coords(patch_size, patch_size, CV_16SC2);
		Mat patch;
		for (int i = 0; i < roi.height; ++i) {
			for (int j = 0; j < roi.width; ++j) {
				coords.at<Vec2s>(i, j) = Vec2s(roi.x + j, roi.y + i);
			}
		}
		remap(source, patch, coords, Mat(), INTER_NEAREST, BORDER_CONSTANT);

		Point ps(patch_size, patch_size);
		circle(patch, ps / 2, patch_size / 2 - 3, Scalar(255, 0, 0), 1, CV_AA);
		for (auto cp : fp.controlpoints) {
			line(patch, ps / 2, ps / 2 + Point(cp), Scalar(0, 0, 255), 1, CV_AA);
		}

		return patch;
	}
}