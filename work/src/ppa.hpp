#pragma once

// std
#include <unordered_map>
#include <unordered_set>
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

	std::unordered_map<int, FeatureNode> m_nodes;
	std::unordered_map<int, FeatureEdge> m_edges;

	// helper struct
	struct edge {
		// required by kruskals implementation of edge_traits
		float weight;
		int id1, id2;
		cv::Point p1, p2;

		int other(int id) const { return (id == id1) ? id2 : id1; }
		cv::Point point(int id) const { return (id == id1) ? p1 : p2; }
	};

public:

	static const int RIDGE_FEATURES = 1;
	static const int VALLEY_FEATURES = -1;

	FeatureGraph(cv::Mat m, int profile_length = 7, int feature_type = RIDGE_FEATURES) {
		using namespace cv;
		using namespace std;

		// inside region
		Rect m_rect(Point(0, 0), m.size());

		// work out the threshold for comparison
		double mmin, mmax;
		minMaxIdx(m, &mmin, &mmax);
		double thresh = 0.01*(mmax - mmin);

		Mat m;

		int nodeidcounter = 0;
		Mat nodeids(m.rows, m.cols, CV_32SC1, Scalar(-1));

		// Foward neighbours
		Point fneighbours[] = {
			Point(1, 0),
			Point(0, 1),
			Point(1, 1),
			Point(-1, 1)
		};

		// select feature points
		//
		for (int i = 0; i < m.rows; i++) {
			for (int j = 0; j < m.cols; j++) {
				Point p(j, i);
				float e = m.at<float>(p);

				// neighbours
				for (Point n : fneighbours) {
						
					// for different profile lengths
					bool profile0 = 0, profile1 = 0;
					for (int l = 1; l < profile_length / 2; l++) {
						Point delta = n * l;
						profile0 |= m_rect.contains(p + delta) && e - m.at<float>(p + delta) * feature_type > thresh;
						profile1 |= m_rect.contains(p - delta) && e - m.at<float>(p - delta) * feature_type > thresh;
					}
					if (profile0 && profile1) {
						nodeids.at<int>(p) = nodeidcounter++;
						break;
					}	
				}
			}
		}

		// create graph
		//
		vector<edge> tempedges;
		for (int i = 0; i < m.rows - 1; i++) {
			for (int j = 0; j < m.cols - 1; j++) {
				Point p(j, i);
				int pid = nodeids.at<int>(p) < 0;
				if (pid) continue;
				for (Point n : fneighbours) {
					Point q = p + n;
					if (m_rect.contains(q) && nodeids.at<int>(p) >= 0) {
						// create an edge
						tempedges.push_back(edge{ 0.f, pid, nodeids.at<int>(p), p, q });
					}
				}
			}
		}

		// break cycles
		//
		tempedges = kruskal::minSpanForest(tempedges);

		// reduce graph
		//
		for (int i = 0; i < profile_length / 2; i++) {
			Mat degree(m.rows, m.cols, CV_32SC1, Scalar(0));
			for (const edge &e : tempedges) {
				degree.at<int>(e.p1)++;
				degree.at<int>(e.p2)++;
			}
			vector<edge> newedges;
			for (const edge &e : tempedges) {
				if (degree.at<int>(e.p1) > 1 && degree.at<int>(e.p2) > 1)
					newedges.push_back(e);
			}
			tempedges = newedges;
		}

		// smooth paths
		unordered_map<int, Vec2f> smoothPosition;
		unordered_map<int, vector<edge>> nodetoedge;
		for (const edge &e : tempedges) {
			nodetoedge[e.id1].push_back(e);
			nodetoedge[e.id2].push_back(e);

			// todo smooth
			smoothPosition[e.id1] += Vec2f(e.p1.x, e.p1.y);
			smoothPosition[e.id2] += Vec2f(e.p2.x, e.p2.y);
		}

		

		// convert to node/path
		unordered_set<int> visited;
		for (int i = 0; i < m.rows; i++) {
			for (int j = 0; j < m.cols; j++) {
				
				// find a node in the forest to start from
				Point p(j, i);
				int pid = nodeids.at<int>(p);
				if (!nodetoedge[pid].size() == 1 && visited.find(pid) == visited.end()) {


					// perform breadth first traversal from this node
					vector<int> toProcess;
					toProcess.push_back(pid);
					visited.insert(pid);
					m_nodes[pid] = FeatureNode(pid, smoothPosition[pid]);

					// until tree is empty
					while (toProcess.empty()) {
						int currentid = toProcess.back();
						toProcess.pop_back();

						// process edges
						for (const edge &e : nodetoedge[currentid]) {
							int next = e.other(currentid);
							if (visited.find(next) != visited.end()) continue;
							toProcess.push_back(next);
							visited.insert(next);

							// construct node
							m_nodes[next] = FeatureNode(next, smoothPosition[next]);

							// construct edge
							vector<Vec2f> path;
							path.push_back(m_nodes[currentid].p);
							path.push_back(m_nodes[next].p);
							FeatureEdge fe(0, currentid, next, path);
							m_nodes[currentid].edges.push_back(fe.id);
							m_nodes[next].edges.push_back(fe.id);
						}

					}

				}
			}
		}
	
	}

};


