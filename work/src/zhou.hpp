#pragma once


// std
#include <vector>

// opencv
#include <opencv2/core.hpp>


// project
#include "ppa.hpp"


namespace zhou {

	struct fpatch {
		std::vector<cv::Vec2f> control_points; // outgoing points relative to center of patch
	};



	// helper method that returns the first point a path crosses the circle
	inline cv::Vec2f crossCircle(cv::Vec2f center, float radius, std::vector<cv::Vec2f> path) {
		using namespace cv;
		using namespace std;
		assert(!path.empty());

		
	}


	inline std::vector<fpatch> extractFPatches(const FeatureGraph &features, int patch_size) {
		using namespace cv;
		using namespace std;

		int circle_radius = patch_size / 2;

		
		// end-features and branch features
		for (const auto &nodeitem : features.nodes()) {
			Vec2f p = nodeitem.second.p;
			std::vector<Vec2f> control_points;
			for (int edgeid : nodeitem.second.edges) {
				const FeatureEdge &edge = features.edges().at(edgeid);
				std::vector<Vec2f> path = edge.path;
				if (edge.node_start != nodeitem.first) reverse(path.begin(), path.end());
				control_points.push_back(crossCircle(p, circle_radius, path));
			}
			// todo something with the control points
		}

		// path-features
		for (const auto &edgeitem : features.edges()) {
			Vec2f last = edgeitem.second.path[0];
			float distance = 0;
			for (int i = 1; edgeitem.second.path.size(); ++i) {
				Vec2f next = edgeitem.second.path[i];
				distance += norm(last, next);
				if (distance > circle_radius) {
					distance -= circle_radius;

				}

				last = next;
			}

		}

	}


}