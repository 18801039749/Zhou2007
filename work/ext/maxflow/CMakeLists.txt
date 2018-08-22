add_library(maxflow
	graph.cpp
	maxflow.cpp
)

target_include_directories(maxflow PUBLIC "${PROJECT_SOURCE_DIR}/ext/maxflow/include")