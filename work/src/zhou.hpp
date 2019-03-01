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
		int nonfeatureSpacing = 100;

		// feature patch
		float tpsWeight = 1000;
		float featureGraphcutWeight = 1;
		float featureProfileWeight = 3;
		float featureProfileCount = 7;

		// non-feature patch
		int k_set = 3;
		float nonfeatureOverlapWeight = 1;
		float nonfeatureGraphcutWeight = 1;


		// alternative algorithm
		enum {
			PATHPATCH_TPS,
			PATHPATCH_CORNER_TPS,
			PATHPATCH_ROTATE
		};
		int pathPatchAlgorithm = PATHPATCH_ROTATE;

	};

	struct featurePatchCandidate{
		fpatch fp;
		float weight;
		cv::Mat patch;
		cv::Mat graphcut;
	};

	struct nonfeaturePatchCandidate {
		float weight;
		cv::Mat patch;
		cv::Mat graphcut;
	};

	struct nonfeaturePatchTarget {
		int overlappingPixels;
		cv::Vec2i position; // topleft
		cv::Mat patch;
	};


	featurePatchCandidate createFeaturePatchCandidate(const cv::Mat examplemap, const cv::Mat synthesis, fpatch candidate, fpatch target, synthesisparams params) {
		assert(!examplemap.empty());
		assert(!synthesis.empty());
		assert(examplemap.type() == CV_32FC1);
		assert(synthesis.type() == CV_32FC1);

		using namespace cv;
		using namespace std;

		int hs1 = params.patchsize / 2;
		int hs2 = params.patchsize - hs1;
		Vec2f patchCenter(hs1, hs1);

		float cost = 0;

		featurePatchCandidate cand;

		// sample patch
		//
		Mat patchCoords(params.patchsize, params.patchsize, CV_32FC2);
		// if the degree doesn't match, just copy the patch directly
		if (candidate.controlpoints.size() != target.controlpoints.size()) {
			for (int i = 0; i < params.patchsize; ++i) {
				for (int j = 0; j < params.patchsize; ++j) {
					Vec2f p = Vec2f(j, i) - patchCenter + candidate.center;
					patchCoords.at<Vec2f>(i, j) = p;
				}
			}
		}
		// if the degree is 1, use rotation
		else if (candidate.controlpoints.size() == 1) {
			Vec2f candidateAlign = normalize(candidate.controlpoints[0]);
			Vec2f targetAlign = normalize(target.controlpoints[0]);

			// calculate the rotation from candidate to the target
			float sign = copysign(1, targetAlign[0] * candidateAlign[1] - targetAlign[1] * candidateAlign[0]);
			float angle = sign * acos(targetAlign.dot(candidateAlign));
			float c = cos(angle);
			float s = sin(angle);

			// apply the rotation
			for (int i = 0; i < params.patchsize; ++i) {
				for (int j = 0; j < params.patchsize; ++j) {
					Vec2f p = Vec2f(j, i) - patchCenter;
					p = Vec2f(c * p[0] - s * p[1], s * p[0] + c * p[1]);
					p += candidate.center;
					patchCoords.at<Vec2f>(i, j) = p;
				}
			}
		}
		// otherwise degrees are the same, use thin-plate splines
		else {

			// rotation alternative
			if (params.pathPatchAlgorithm == params.PATHPATCH_ROTATE && target.controlpoints.size() == 2) {

				// calculate the average outpath for the candidate
				Vec2f candidateOut0 = normalize(candidate.controlpoints[0]);
				Vec2f candidateOut1 = normalize(candidate.controlpoints[1]);
				Vec2f candidateAlign = (candidateOut0 + candidateOut1) / 2;
				if (isnan(norm(candidateAlign))) { candidateAlign = Vec2f(candidateOut0[1], candidateOut0[0]); }
				else { candidateAlign = normalize(candidateAlign); }

				// calculate the average outpath for the target
				Vec2f targetOut0 = normalize(target.controlpoints[0]);
				Vec2f targetOut1 = normalize(target.controlpoints[1]);
				Vec2f targetAlign = (targetOut0 + targetOut1) / 2;
				if (isnan(norm(targetAlign))) { targetAlign = Vec2f(targetOut0[1], targetOut0[0]); }
				else { targetAlign = normalize(targetAlign); }


				// calculate the rotation from candidate to the target
				float sign = copysign(1, targetAlign[0] * candidateAlign[1] - targetAlign[1] * candidateAlign[0]);
				float angle = sign * acos(targetAlign.dot(candidateAlign));
				float c = cos(angle);
				float s = sin(angle);

				// apply the rotation
				for (int i = 0; i < params.patchsize; ++i) {
					for (int j = 0; j < params.patchsize; ++j) {
						Vec2f p = Vec2f(j, i) - patchCenter;
						p = Vec2f(c * p[0] - s * p[1], s * p[0] + c * p[1]);
						p += candidate.center;
						patchCoords.at<Vec2f>(i, j) = p;
					}
				}
			}
			else {
				// COST of spline
				//
				thinplate2d<float> bestspline;
				for (int offset = 0; offset < target.controlpoints.size(); offset++) {
					thinplate2d<float> spline;
					spline.addPoint(target.center, candidate.center); // center
					for (int n = 0; n < target.controlpoints.size(); ++n) { // outpaths
						int idx = (n + offset) % target.controlpoints.size();
						spline.addPoint(target.controlpoints[n] + target.center, candidate.controlpoints[n] + candidate.center);
					}
					// corners
					if (params.pathPatchAlgorithm == params.PATHPATCH_CORNER_TPS) {
						spline.addPoint(target.center + Vec2f( hs1,  hs1), candidate.center + Vec2f( hs1,  hs1));
						spline.addPoint(target.center + Vec2f( hs1, -hs1), candidate.center + Vec2f( hs1, -hs1));
						spline.addPoint(target.center + Vec2f(-hs1,  hs1), candidate.center + Vec2f(-hs1,  hs1));
						spline.addPoint(target.center + Vec2f(-hs1, -hs1), candidate.center + Vec2f(-hs1, -hs1));
					}

					// essential
					spline.computeWeights();

					if (spline.energy() < bestspline.energy()) {
						bestspline = spline;
					}
				}

				for (int i = 0; i < params.patchsize; ++i) {
					for (int j = 0; j < params.patchsize; ++j) {
						Vec2f p = Vec2f(j, i) - patchCenter + target.center;
						patchCoords.at<Vec2f>(i, j) = bestspline.evaluate(p);
					}
				}
				cost += bestspline.energy() * params.tpsWeight; // should be close to zero for 3 points
			}


	
		}

		// create patch (making sure we don't use patches off the example)
		remap(examplemap, cand.patch, patchCoords, Mat(), INTER_LINEAR, BORDER_CONSTANT, Scalar(numeric_limits<float>::quiet_NaN()));
		if (isnan(sum(cand.patch)[0])) {
			cand.weight = numeric_limits<float>::infinity();
			return cand;
		}
		remap(examplemap, cand.patch, patchCoords, Mat(), INTER_LINEAR, BORDER_REPLICATE);

		
		// COST of ridge profile 
		//
		if (candidate.controlpoints.size() == target.controlpoints.size() && !target.controlpoints.empty()) {
			float ridgeSSD = 0;
			const int controlPoints = target.controlpoints.size();
			const int profilePoints = params.featureProfileCount;

			Mat targetCoords(controlPoints, profilePoints, CV_32FC2);
			Mat patchCoords(controlPoints, profilePoints, CV_32FC2);

			// for each outpatch (control point)
			for (int n = 0; n < controlPoints; ++n) {

				Vec2f targetOutpath = target.center + target.controlpoints[n];
				Vec2f patchOutpath = patchCenter + target.controlpoints[n];
				Vec2f perpendicular = normalize(Vec2f(target.controlpoints[n][1], target.controlpoints[n][0]));

				//cout << "targetCenter: " << targetCenter << endl;
				//cout << "targetOutpath: " << targetOutpath << endl;
				//cout << "patchCenter: " << patchCenter << endl;
				//cout << "patchOutpath: " << patchOutpath << endl;
				//cout << "perpendicular: " << targetCenter << endl;

				// sample perpendicular line across the outpath of the synthesis and the patch
				for (int p = 0; p < profilePoints; ++p) {
					float distance = p - float(profilePoints) / 2;
					targetCoords.at<Vec2f>(n, p) = distance * perpendicular + targetOutpath;
					patchCoords.at<Vec2f>(n, p) = distance * perpendicular + patchOutpath;
				}
			}

			// remap and don't consider nan values
			Mat targetRidge, patchRidge;
			remap(synthesis, targetRidge, targetCoords, Mat(), CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(numeric_limits<float>::quiet_NaN()));
			remap(cand.patch, patchRidge, patchCoords, Mat(), CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(numeric_limits<float>::quiet_NaN()));

			// ridge difference
			for (int n = 0; n < controlPoints; ++n) {
				for (int p = 0; p < profilePoints; ++p) {
					float d = targetRidge.at<float>(n, p) - patchRidge.at<float>(n, p);
					if (!isnan(d)) {
						ridgeSSD += d * d;
					}
				}
			}

			cost += ridgeSSD * params.featureProfileWeight;
		}


		// COST of graphcut
		//
		float graphcut_cost;
		cand.graphcut = zhou::graphcut(synthesis, cand.patch, Vec2i(target.center - patchCenter), &graphcut_cost);
		cost += graphcut_cost * params.featureGraphcutWeight;


		// finished
		cand.weight = cost;
		return cand;
	}



	inline nonfeaturePatchCandidate createNonfeaturePatchCandidate(cv::Mat candidate, cv::Mat target, synthesisparams params) {
		assert(!candidate.empty());
		assert(!target.empty());
		assert(candidate.type() == CV_32FC1);
		assert(target.type() == CV_32FC1);
		assert(candidate.size() == target.size());

		using namespace cv;
		using namespace std;

		nonfeaturePatchCandidate cand;
		cand.patch = candidate;
		float cost = 0;

		// COST of graphcut
		//
		float graphcut_cost;
		cand.graphcut = zhou::graphcut(target, cand.patch, Vec2i(0, 0), &graphcut_cost);
		cost += graphcut_cost * params.featureGraphcutWeight;


		// COST of SSD
		//
		float ssd = 0;
		for (int i = 0; i < params.patchsize; ++i) {
			for (int j = 0; j < params.patchsize; ++j) {
				float d = candidate.at<float>(i, j) - target.at<float>(i, j);
				if (!isnan(d)) {
					ssd += d * d;
				}
			}
		}
		cost += ssd * params.nonfeatureOverlapWeight;


		// finished
		cand.weight = cost;
		return cand;
	}




	inline cv::Mat synthesize(const cv::Mat examplemap, const cv::Mat sketchmap, synthesisparams params) {
		assert(examplemap.type() == CV_32FC1);
		assert(sketchmap.type() == CV_32FC1);

		using namespace cv;
		using namespace std;

		// unsynthesized regions are marked with NaN
		Mat synthesis(sketchmap.rows, sketchmap.cols, CV_32FC1, Scalar(numeric_limits<float>::quiet_NaN()));
		int hs1 = params.patchsize / 2;

		// TODO for ridge and valley seperately
		// 1) Identify features
		//
		ppa::FeatureGraph examplefeature(examplemap);
		ppa::FeatureGraph sketchfeature(sketchmap);

		// 2) Extract feature patches
		//
		vector<fpatch> featurepatches = extractFeaturePatches(examplefeature, params.patchsize);

		// 3) Place feature patches
		//
		for (fpatch target : extractFeaturePatches(sketchmap, params.patchsize)) {
			// calculate primary costs for relevant patches
			priority_queue<featurePatchCandidate> candidates;
			featurePatchCandidate best;
			best.weight = numeric_limits<float>::infinity();
			for (fpatch candidate : featurepatches) {
				// find the best matching feature patch
				if (candidate.controlpoints.size() == target.controlpoints.size()) {
					featurePatchCandidate cand = createFeaturePatchCandidate(examplemap, synthesis, candidate, target, params);
					if (cand.weight < best.weight) {
						best = cand;
					}
				}
			}

			// if we didn't find a matching candidate, use a non-matching candidate
			if (isinf(best.weight)) {
				for (fpatch candidate : featurepatches) {
					if (candidate.controlpoints.size() != target.controlpoints.size()) {
						featurePatchCandidate cand = createFeaturePatchCandidate(examplemap, synthesis, candidate, target, params);
						if (cand.weight < best.weight) {
							best = cand;
						}
					}
				}
			}

			//cout << best.weight << endl;
			//imwrite("output/patch.png", best.patch);
			//imwrite("output/synthesis.png", synthesis);
			//Mat gc(best.graphcut.size(), CV_8UC1);
			//for (int i = 0; i < gc.rows; ++i) {
			//	for (int j = 0; j < gc.cols; ++j) {
			//		gc.at<uchar>(i, j) = (best.graphcut.at<uchar>(i, j) ? 255 : 0);
			//	}
			//}
			//imwrite("output/graphcut.png", gc);

			// place patch
			zhou::placePatch(synthesis, best.patch, best.graphcut, Vec2i(target.center[0] - hs1, target.center[1] - hs1));
			imwrite("output/synthesis.png", synthesis);
		}


		// 4) Extract and place non-feature patches
		//
		vector<Mat> nonfeaturePatches = extractNonfeaturePatches(examplemap, featurepatches, params.patchsize);

		auto cmp = [](const nonfeaturePatchTarget &left, const nonfeaturePatchTarget &right) { return left.overlappingPixels > right.overlappingPixels; };
		priority_queue<nonfeaturePatchTarget, vector<nonfeaturePatchTarget>, decltype(cmp)> targetPatches(cmp);

		for (int offset = 0; offset < params.nonfeatureSpacing; offset += params.patchsize/2) {
			for (int y = -hs1; y < synthesis.rows; y += params.nonfeatureSpacing) {
				for (int x = -hs1; x < synthesis.cols; x += params.nonfeatureSpacing) {

					nonfeaturePatchTarget target;

					// set position
					target.position = Vec2i(offset + x, offset + y);

					Mat targetCoords(params.patchsize, params.patchsize, CV_16SC2);
					for (int i = 0; i < params.patchsize; ++i) {
						for (int j = 0; j < params.patchsize; ++j) {
							targetCoords.at<Vec2s>(i, j) = Vec2s(offset + x + j, offset + y + i);
						}
					}
					remap(synthesis, target.patch, targetCoords, Mat(), INTER_NEAREST, BORDER_CONSTANT, Scalar(numeric_limits<float>::quiet_NaN()));

					// count overlapping values
					target.overlappingPixels = 0;
					for (int i = 0; i < params.patchsize; ++i) {
						for (int j = 0; j < params.patchsize; ++j) {
							if (isnan(target.patch.at<float>(i, j)))
								target.overlappingPixels++;
						}
					}

					

					targetPatches.push(target);
				}
			}


			// synthesize non-feature patches
			while (!targetPatches.empty()) {
				nonfeaturePatchTarget target = targetPatches.top();
				targetPatches.pop();

				// find the best candidate
				nonfeaturePatchCandidate best;
				best.weight = numeric_limits<float>::infinity();
				for (Mat candidate : nonfeaturePatches) {
					nonfeaturePatchCandidate cand = createNonfeaturePatchCandidate(candidate, target.patch, params);
					if (cand.weight < best.weight) {
						best = cand;
					}
				}

				zhou::placePatch(synthesis, best.patch, best.graphcut, target.position);
				imwrite("output/synthesis.png", synthesis);
			}
		}


		return synthesis;
	}

}