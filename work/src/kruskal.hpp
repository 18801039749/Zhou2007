

#include <queue>
#include <iostream>


namespace kruskal {

	//class implementing Union Find Data Structure with Path Compression
	class UnionFind {
	  public:

		std::vector<int> id;
		std::vector<int> sz;

		UnionFind(int n) : id(n), sz(n, 1) {
			for (int i = 0; i < n; ++i) {
				id[i] = i;
			}
		}
		
		int root(int i) {
			while(i != id[i]) {
				id[i] = id[id[i]];	//path Compression
				i = id[i];
			}
			return i;
		}

		int find(int p, int q) {
			return root(p)==root(q);
		}

		void unite(int p, int q) {
			int i = root(p);
			int j = root(q);

			if(sz[i] < sz[j]) {
				id[i] = j;
				sz[j] += sz[i];
			} else {
				id[j] = i;
				sz[i] += sz[j];
			}
		}
	};

	template <typename T>
	struct edge_traits {
		static int id1(const T &e) { return e.id1; }
		static int id2(const T &e) { return e.id2; }
		static float weight(const T &e) { return e.weight; }
	};


	template <typename E, typename T = edge_traits<E>>
	std::vector<E> minSpanForest(std::vector<E> edges) {

		using namespace std;

		// priority queue of edges
		auto comp = [](E &a, E &b) { return T::weight(a) < T::weight(b); };
		std::priority_queue<E, std::vector<E>, decltype(comp)> remaining_edges(comp);
		int max_id = 0;
		for (E e : edges) {
			remaining_edges.push(e);
			if (T::id1(e) > max_id) max_id = T::id1(e);
			if (T::id2(e) > max_id) max_id = T::id2(e);
		}

		// create the unionfind and return vector
		UnionFind uf_set(max_id + 1); // max_id+1 for index offset
		std::vector<E> minimum_forest;

		// perform kruskals
		while (!remaining_edges.empty()) {
			E e = remaining_edges.top();
			remaining_edges.pop();

			if (!uf_set.find(T::id1(e), T::id2(e))) {
				minimum_forest.push_back(e);
				uf_set.unite(T::id1(e), T::id2(e));
			}
		}


		return minimum_forest;
	}


}
