#pragma once

// std
#include <vector>
#include <queue>

// opencv
#include <opencv2/core.hpp>

// project
#include "ppa.hpp"
#include "featurepatch.hpp"

namespace zhou {

	struct synthesisparams {
		bool ridges = false;
		bool valleys = false;
		int patchsize = 80;
		int k_set = 3;

		// TODO weights

	};

	struct fpatch_candidate{
		fpatch fp;
		float weight;
		// TODO other precomputed stuff
		// like the heightmap calculated on the fly
	};

	inline cv::Mat synthesize(const cv::Mat examplemap, const cv::Mat sketchmap, synthesisparams params) {
		using namespace cv;
		using namespace std;

		assert(examplemap.type() == CV_32FC1);
		assert(sketchmap.type() == CV_32FC1);

		Mat synthesized(sketchmap.rows, sketchmap.cols, CV_8UC1, Scalar(0));
		Mat synthesis(sketchmap.rows, sketchmap.cols, CV_32FC1, Scalar(0));

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
				for (fpatch candidate : featurepatches) {
					// TODO calculate fpatch_candidate
				}

				// calculate secondary costs for best k-set 
				fpatch_candidate best;
				for (int i = 0; i < 5 && !candidates.empty(); ++i) {
					fpatch_candidate current = candidates.top;
					candidates.pop();

					// TODO calculate graphcut cost?
					if (current.weight < best.weight)
						best = current;
				}

				// place "best" patch
				// TODO place "best" patch 
			}
		}


		// 4) Extract and place non-feature patches
		//

		// ??? how is this even done ???


		return synthesis;
	}

}