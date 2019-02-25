#pragma once

// std
#include <vector>
#include <queue>

// opencv
#include <opencv2/core.hpp>

// project
#include "ppa.hpp"
#include "thin_plate.hpp"
#include "featurepatch.hpp"
#include "graphcut.hpp"
#include "patchmerge.hpp"

namespace zhou {

	struct synthesisparams {
		bool ridges = false;
		bool valleys = false;
		int patchsize = 80;

		// feature patch
		float tps_weight = 1;
		float feature_weight = 1;
		float fgraphcut_weight = 1;

		// non-feature patch
		int k_set = 3;
		float overlap_weight = 1;
		float nfgraphcut_weight = 1;

	};

	struct fpatch_candidate{
		fpatch fp;
		float weight;
		cv::Mat patch;
		cv::Mat graphcut;
		// TODO other precomputed stuff
		// and some measure
	};




	fpatch_candidate createFeaturePatchCandidate(const cv::Mat examplemap, const cv::Mat synthesis, fpatch candidate, fpatch target, synthesisparams params) {
		// TODO asserts

		using namespace cv;
		using namespace std;

		int hs1 = params.patchsize / 2;
		int hs2 = params.patchsize - hs1;

		float cost = 0;

		fpatch_candidate cand;

		// sample patch
		//
		Mat samplecoord(params.patchsize, params.patchsize, CV_32FC2);
		// if the degree doesn't match, just copy the patch directly
		if (target.controlpoints.size() != target.controlpoints.size()) {
			for (int i = 0; i < params.patchsize; ++i) {
				for (int j = 0; j < params.patchsize; ++j) {
					Vec2f p = Vec2f(j, i) - Vec2f(hs1, hs1) + candidate.center;
					samplecoord.at<Vec2f>(i, j) = p;
				}
			}
		}
		// if the degree is 1, use rotation
		else if (candidate.controlpoints.size() == 1) {
			// rotate
			float angle = atan2(candidate.controlpoints[0][1], candidate.controlpoints[0][0]);
			float c = cos(angle);
			float s = sin(angle);

			for (int i = 0; i < params.patchsize; ++i) {
				for (int j = 0; j < params.patchsize; ++j) {
					Vec2f p = Vec2f(j, i) - Vec2f(hs1, hs1);
					p = Vec2f(c * p[0] + s * p[1], c * p[0] - s * p[1]);
					p += candidate.center;
					samplecoord.at<Vec2f>(i, j) = p;
				}
			}
		}
		// otherwise, use thin-plate splines
		else {
			thinplate2d<float> spline;
			spline.addPoint(target.center, candidate.center);
			for (int n = 0; n < target.controlpoints.size(); ++n) {
				spline.addPoint(target.controlpoints[n], candidate.controlpoints[n]);
			}
			for (int i = 0; i < params.patchsize; ++i) {
				for (int j = 0; j < params.patchsize; ++j) {
					Vec2f p = Vec2f(j, i) - Vec2f(hs1, hs1) + target.center;
					samplecoord.at<Vec2f>(i, j) = spline.evaluate(p);
				}
			}
			cand.weight += spline.energy() * params.tps_weight;
		}

		// create patch
		//
		remap(examplemap, cand.patch, samplecoord, Mat(), INTER_LINEAR, BORDER_REPLICATE);

		// TODO
		// cost of ridge profile SSD
		//
		cand.weight += 0;


		// graph cut
		//
		float graphcut_cost;
		cand.graphcut = zhou::graphcut(synthesis, cand.patch, Vec2i(target.center[0] - hs1, target.center[1] - hs1), &graphcut_cost);
		cand.weight += graphcut_cost;

		//
		//
		return cand;
	}




	inline cv::Mat synthesize(const cv::Mat examplemap, const cv::Mat sketchmap, synthesisparams params) {
		using namespace cv;
		using namespace std;

		assert(examplemap.type() == CV_32FC1);
		assert(sketchmap.type() == CV_32FC1);

		// unsynthesized regions are marked with NaN
		Mat synthesis(sketchmap.rows, sketchmap.cols, CV_32FC1, Scalar(numeric_limits<float>::quiet_NaN()));

		// for ridge and valley seperately
		{
			// 1) Identify features
			//
			ppa::FeatureGraph examplefeature(examplemap);
			ppa::FeatureGraph sketchfeature(sketchmap);

			// 2) Extract feature patches
			//
			vector<fpatch> featurepatches = extractFeaturePatches(examplefeature, params.patchsize);

			// 3) Place feature patches
			//
			for (fpatch fp : extractFeaturePatches(sketchmap, params.patchsize)) {
				// calculate primary costs for relevant patches
				priority_queue<fpatch_candidate> candidates;
				fpatch_candidate best;
				for (fpatch candidate : featurepatches) {
					fpatch_candidate cand = createFeaturePatchCandidate(examplemap, synthesis, candidate, fp, params);
					if (cand.weight < best.weight) {
						best = cand;
					}
				}

				// place patch
				zhou::placePatch(synthesis, best.patch, best.graphcut, fp.center);
			}
		}


		// 4) Extract and place non-feature patches
		//

		// ??? how is this even done ???


		return synthesis;
	}

}