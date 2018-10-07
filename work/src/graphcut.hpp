#pragma once

// std
#include <iostream>

// maxflow
#include <maxflow/graph.h>

// opencv
#include <opencv2/core.hpp>

namespace zhou {

	inline void print_graphcut_error(const char *c) {
		std::cerr << c << std::endl;
	}

	// the synthesis is a patch sized segment of the terrain synthesis that the patch will be placed on
	// the patch itself must not have any NaN values
	// returns a mask of the cut
	inline cv::Mat graphcut(cv::Mat synthesis, cv::Mat patch, float *cost = nullptr) {
		using FloatGraph_t = Graph<float, float, float>;
		using namespace std;
		using namespace cv;

		assert(synthesis.size() == patch.size());
		assert(synthesis.type() == CV_32FC1);
		assert(patch.type() == CV_32FC1);

		// create graphcut patch, graph and array
		const float max_edge = 1e10; // numeric_limits<float>::max();
		Mat patch_cut(synthesis.rows, synthesis.cols, CV_8UC1, true);
		Mat area_id(synthesis.rows, synthesis.cols, CV_32SC1, Scalar(-1)); // stores node ids for each point
		FloatGraph_t graph(synthesis.cols * synthesis.rows, synthesis.cols * synthesis.rows * 4, print_graphcut_error);
		
		Mat ids(synthesis.rows, synthesis.cols, CV_8UC3, Scalar(0, 0, 0));
		Mat debug(synthesis.rows, synthesis.cols, CV_8UC3, Scalar(0, 0, 0));

		// generate nodes for every (non-NaN) point
		for (int i = 0; i < synthesis.rows; ++i) {
			for (int j = 0; j < synthesis.cols; ++j) {
				if (!isnan(synthesis.at<float>(i, j))) {
					ids.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
					area_id.at<int>(i, j) = graph.add_node();
				}
			}
		}

		// right, up, left, down
		const Point delta[] = { Point{1, 0}, Point{0, 1}, Point{-1, 0}, Point{0, -1} };

		// source/sink count
		int source_count = 0;
		int sink_count = 0;

		// connect nodes
		Rect area(Point(0, 0), synthesis.size());
		Mat diff = abs(synthesis - patch);
		for (int i = 0; i < synthesis.rows; ++i) {
			for (int j = 0; j < synthesis.cols; ++j) {
				Point p(j, i);

				// if this is synthesis node in the graph
				if (area_id.at<int>(p) >= 0) {
					bool in_sink = false;
					bool in_source = false;

					// for every neighbour
					for (int d = 0; d < 4; ++d) {
						Point q = p + delta[d]; // neighbour point on the area, in area space

						// if neighbour is inside the area
						if (area.contains(q)) {
							// if neighbour is valid node connect the nodes using a graph edge
							if (area_id.at<int>(q) >= 0) {
								float val = diff.at<float>(p) + diff.at<float>(q);
								if (isnan(val)) throw runtime_error("NaN value in graphcut");
								graph.add_edge(area_id.at<int>(p), area_id.at<int>(q), val, 0);
							}
							// otherwise connect to the sink
							else { 
								in_sink = true;
							}
						}
						// on the edge, connect to the source
						else {
							in_source = true;
						}
					}

					// center of the patch is always a sink
					if (p.x == synthesis.cols / 2 && p.y == synthesis.rows / 2) {
						in_sink = true;
					}

					// connect to one of, source or sink
					if (in_source) {
						ids.at<Vec3b>(p) = Vec3b(255, 0, 0);
						graph.add_tweights(area_id.at<int>(p), max_edge, 0);
						++source_count;
					}
					else if (in_sink) {
						ids.at<Vec3b>(p) = Vec3b(0, 255, 0);
						graph.add_tweights(area_id.at<int>(p), 0, max_edge);
						++sink_count;
					}
					else {
					}

				}
			}
		}

		imwrite("output/ids.png", ids);


		// if there are no sources or no sinks we return the patch as is
		if (source_count == 0 || sink_count == 0) {
			return patch_cut;
		}

		// compute the maxflow/mincut
		float c = graph.maxflow();
		if (cost != nullptr) {
			*cost = c;
		}

		// for every point in the area
		for (int i = 0; i < synthesis.rows; ++i) {
			for (int j = 0; j < synthesis.cols; ++j) {
				Point p(j, i);
				int id = area_id.at<int>(p);
				// if there was no node or connected to the source
				patch_cut.at<uchar>(p) = !(id >= 0 && graph.what_segment(id) == FloatGraph_t::SOURCE);

				if (id >= 0) {
					debug.at<Vec3b>(p) = Vec3b(0, 0, 255);
					if (graph.what_segment(id) == FloatGraph_t::SOURCE)
						debug.at<Vec3b>(p) = Vec3b(255, 0, 0);
					if (graph.what_segment(id) == FloatGraph_t::SINK)
						debug.at<Vec3b>(p) = Vec3b(0, 255, 0);
				}
			}
		}

		imwrite("output/debug.png", debug);
		imwrite("output/diff.png", diff);

		return patch_cut;
	}


	// returns a mask of the cut relative to the patch size
	inline cv::Mat graphcut(cv::Mat synthesis, cv::Mat patch, cv::Vec2i pos, float *cost = nullptr) {
		using namespace std;
		using namespace cv;

		assert(synthesis.type() == CV_32FC1);
		assert(patch.type() == CV_32FC1);

		Vec2i ps(patch.cols, patch.rows);
		
		Mat bsynthesis;
		copyMakeBorder(synthesis, bsynthesis, ps[1], ps[1], ps[0], ps[0], BORDER_CONSTANT, numeric_limits<double>::quiet_NaN());

		imwrite("output/isnan.png", bsynthesis);

		int startcol = pos[0] + ps[0];
		int startrow = pos[1] + ps[1];
		Mat synthesis_patch = bsynthesis(Range(startrow, startrow + ps[1]), Range(startcol, startcol + ps[0]));

		return graphcut(synthesis_patch, patch, cost);
	}
}