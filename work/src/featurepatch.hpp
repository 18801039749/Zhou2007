#pragma once

// std
#include <vector>

// opencv
#include <opencv2/core.hpp>

// project
#include "ppa.hpp"


namespace zhou {

	struct fpatch {
		cv::Vec2f center;
		std::vector<cv::Vec2f> controlpoints; // outgoing points relative to center of patch
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
	inline cv::Vec2f circlePathIntersection(cv::Vec2f center, float radius, std::vector<cv::Vec2f> path, bool reversePath = false) {
		using namespace cv;
		using namespace std;
		assert(path.size() >= 2);

		if (reversePath) reverse(path.begin(), path.end());

		Vec2f intersect;
		for (int i = 0; i < path.size() - 1; ++i) {
			if (circleLineIntersection(center, radius, path[i], path[i+1], intersect)) {
				return intersect;
			}
		}

		// no intersection found, return the end of the path extended to the circle
		Vec2f start = path[path.size() - 2];
		Vec2f end = path.back();
		Vec2f d = end - start;
		double n = norm(d);
		end += (d / n) * 2 * radius;
		if (!circleLineIntersection(center, radius, start, end, intersect)) cerr << "ERROR in intersection code" << endl;
		return intersect;
	}

	// TODO return in bredth first order!
	inline std::vector<fpatch> extractFeaturePatches(const ppa::FeatureGraph &features, int patch_size) {
		using namespace cv;
		using namespace std;

		int radius = patch_size / 2;
		vector<fpatch> featurepatches;

		// end-features and branch features
		//
		for (const auto &nodeitem : features.nodes()) {
			Vec2f p = nodeitem.second.p;
			std::vector<Vec2f> controlpoints;
			bool valid = true; // must have all control points
			Vec2f point; // TODO is this correct?
			for (int edgeid : nodeitem.second.edges) {
				const ppa::FeatureEdge &edge = features.edges().at(edgeid);
				std::vector<Vec2f> path = edge.path;
				controlpoints.push_back(circlePathIntersection(p, radius, path, (edge.node_start != nodeitem.first)));
			}

			featurepatches.push_back(fpatch{ p, controlpoints });

		}


		// path-features
		//
		for (const auto &edgeitem : features.edges()) {
			vector<Vec2f> previousPath; // reverse order
			vector<Vec2f> nextPath = edgeitem.second.path;
			previousPath.push_back(nextPath.back());
			nextPath.pop_back();
			float distance = 0;

			while (nextPath.size() >= 2) {
				Vec2f center = nextPath.back();
				distance += norm(previousPath.back(), center);
				previousPath.push_back(center);
				if (distance > radius) {
					distance -= radius;
					// create patch here
					std::vector<Vec2f> controlpoints;
					controlpoints.push_back(circlePathIntersection(center, radius, previousPath, true));
					controlpoints.push_back(circlePathIntersection(center, radius, nextPath, true));
					featurepatches.push_back(fpatch{ center, controlpoints });
				}
				nextPath.pop_back();
			}
		}

		return featurepatches;
	}

}