#pragma once

// std
#include <unordered_map>
#include <vector>

// opencv
#include <opencv2/core.hpp>

// project
#include "kruskal.hpp"



struct FeatureNode {
	int id;
	cv::Vec2f p;
	std::vector<int> edges;
	FeatureNode(int id_, cv::Vec2f p_) : id(id_), p(p_) { }
};

struct FeatureEdge {
	int id;
	int node_start, node_end;
	std::vector<cv::Vec2f> path;
	FeatureEdge(int id_, int ns, int ne, std::vector<cv::Vec2f> path = {})
		: id(id_), node_start(ns), node_end(ne) { }
};


class FeatureGraph {
private:
	int m_node_id_count = 0;
	int m_edge_id_count = 0;

	std::unordered_map<int, FeatureNode> m_nodes;
	std::unordered_map<int, FeatureEdge> m_edges;

	int addNode(const cv::Vec2f &p) {
		m_nodes[m_node_id_count] = FeatureNode(m_node_id_count, p);
		return m_node_id_count++;
	}

	int addEdge(int node_start, int node_end, const std::vector<cv::Vec2f> &path) {
		m_edges[m_edge_id_count] = FeatureEdge(m_edge_id_count, node_start, node_end, path);
		return m_edge_id_count++;
	}

public:

	static const int RIDGE_FEATURES = 1;
	static const int VALLEY_FEATURES = -1;

	FeatureGraph(cv::Mat m, int profile_length = 7, int feature_type = RIDGE_FEATURES) {
		using namespace cv;

		// inside region
		Rect m_rect(Point(0, 0), m.size());

		// work out the threshold for comparison
		double mmin, mmax;
		minMaxIdx(m, &mmin, &mmax);
		double thresh = 0.01*(mmax - mmin);

		Mat m;

		// select feature points
		for (int i = 0; i < m.rows; i++) {
			for (int j = 0; j < m.cols; j++) {
				Point p(j, i);
				float e = m.at<float>(p);

				// neighbours
				for (int ii = -1; ii <= 1; ii++) {
					for (int jj = -1; jj <= 1; jj++) {
						if (ii == 0 && jj == 0) continue;
						

						// for different profile lengths
						for (int l = 1; l < profile_length / 2; l++) {
							Point delta(jj * l, ii * l);
							Point q = p + delta;
							if (m_rect.contains(q) && e - m.at<float>(q) * feature_type > thresh) {
								
							}
						
						}

					}
				}

			}
		}


		// create graph

		// break cycles

		// reduce graph

		// smooth paths
	
	}

};


