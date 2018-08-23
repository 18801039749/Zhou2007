#pragma once

// std
#include <vector>

// eigen
#include <eigen.hpp>
#include <Eigen/Dense>

// opencv
#include <opencv2/core.hpp>


namespace zhou {

	template<typename T>
	class thinplate2d {
	public:
		using VecT = cv::Vec<T, 2>;

	private:
		// data
		std::vector<VecT> m_samples;
		std::vector<VecT> m_values;

		// weights
		std::vector<T> m_weights0;
		std::vector<T> m_weights1;
		cv::Vec<T, 3> m_a0;
		cv::Vec<T, 3> m_a1;

		// energy
		T m_energy;

		// thin plate spline kernel : k(p,q) = d=distance(p,q), d^2 ln(d)
		T thinPlateKernal(const VecT &p, const VecT &q) const {
			double d = cv::norm(p, q);
			T r = 2 * d * d * log(d);
			return (r != r) ? T(0) : r;
		}

	public:
		thinplate2d() { }

		void addPoint(const VecT &sample, const VecT &value) {
			m_samples.push_back(sample);
			m_values.push_back(value);
		}

		void computeWeights() {
			assert(m_samples.size() == m_values.size());
			assert(m_samples.size() >= 3);

			using namespace std;

			// L = [ K    S ]   X = [ W ]   Y = [ V ]
			//     [ S^t  0 ],      [ A ],      [ 0 ]
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> L(m_samples.size() + 3, m_samples.size() + 3);
			Eigen::Matrix<T, Eigen::Dynamic, 1> x(m_samples.size() + 3);
			Eigen::Matrix<T, Eigen::Dynamic, 1> Y0(m_samples.size() + 3);
			Eigen::Matrix<T, Eigen::Dynamic, 1> Y1(m_samples.size() + 3);

			// build L and Y
			// zero out L and Y
			L.setConstant(0);
			Y0.setConstant(0);
			Y1.setConstant(0);
			for (int i = 0; i < m_samples.size(); i++) {

				// K = [ k(s0,s0)  k(s0,s1)  ..  k(s0,sn) ]
				//     [ k(s1,s0)  k(s1,s1)  ..  k(s1,sn) ]
				//     [    ..        ..     ..     ..    ]
				//     [ k(sn,s0)  k(sn,s1)  ..  k(sn,sn) ]
				for (int j = i + 1; j < m_samples.size(); j++) {
					T v = thinPlateKernal(m_samples[i], m_samples[j]);
					L(i, j) = v;
					L(j, i) = v;
				}

				// S = [  1  x0  y0 ]
				//     [  1  x1  y1 ]
				//     [ ..  ..  .. ]
				//     [  1  xn  yn ]
				L(i, m_samples.size() + 0) = 1;
				L(i, m_samples.size() + 1) = m_samples[i][0];
				L(i, m_samples.size() + 2) = m_samples[i][1];

				// S^t
				L(m_samples.size() + 0, i) = 1;
				L(m_samples.size() + 1, i) = m_samples[i][0];
				L(m_samples.size() + 2, i) = m_samples[i][1];

				// V = [ v0 ]
				//     [ v1 ]
				//     [ .. ]
				//     [ vn ]
				Y0(i) = m_values[i][0];
				Y1(i) = m_values[i][1];
			}


			// solve for X
			// Useing LU decomposition because L is symmetric
			Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver(L);
			Eigen::Matrix<T, Eigen::Dynamic, 1> X0 = solver.solve(Y0);
			Eigen::Matrix<T, Eigen::Dynamic, 1> X1 = solver.solve(Y1);

			// W = [ w0 ]
			//     [ w1 ]
			//     [ .. ]
			//     [ wn ]
			m_weights0.clear();
			m_weights0.reserve(m_samples.size());
			m_weights1.clear();
			m_weights1.reserve(m_samples.size());
			for (int i = 0; i < m_samples.size(); i++) {
				m_weights0.push_back(X0(i));
				m_weights1.push_back(X1(i));
			}

			// A = [ a0 ]
			//     [ a1 ]
			//     [ a2 ]
			m_a0 = cv::Vec<T, 3>(X0(m_samples.size() + 0), X0(m_samples.size() + 1), X0(m_samples.size() + 2));
			m_a1 = cv::Vec<T, 3>(X1(m_samples.size() + 0), X1(m_samples.size() + 1), X1(m_samples.size() + 2));

			// caculate the energy I = W^t K W
			Eigen::Matrix<T, Eigen::Dynamic, 1> W0(X0.block(0, 0, m_samples.size(), 1));
			Eigen::Matrix<T, Eigen::Dynamic, 1> W1(X1.block(0, 0, m_samples.size(), 1));
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K(L.block(0, 0, m_samples.size(), m_samples.size()));
			m_energy = W0.transpose() * K * W0;
			m_energy += W1.transpose() * K * W1;

			 //std::cout << "W0" << W0 << std::endl;
			 //std::cout << "a0" << m_a0 << std::endl;
			 //std::cout << "W1" << W1 << std::endl;
			 //std::cout << "a1" << m_a1 << std::endl;
			 //std::cout << "K" << K << std::endl;
			 //std::cout << "m_energy" << m_energy << std::endl << std::endl;
		}


		VecT evaluate(const VecT &p) const {
			VecT result(m_a0[0] + m_a0[1]*p[0] + m_a0[2]*p[1], m_a1[0] + m_a1[1] * p[0] + m_a1[2] * p[1]);
			for (size_t i = 0; i < m_samples.size(); i++) {
				result += VecT(m_weights0[i] * thinPlateKernal(p, m_samples[i]), m_weights1[i] * thinPlateKernal(p, m_samples[i]));
			}
			return result;
		}

		T energy() const {
			return m_energy;
		}


		std::vector<VecT> samples() const {
			return m_samples;
		}
		
		std::vector<VecT> values() const {
			return m_values;
		}
	};


}