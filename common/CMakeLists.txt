add_library(data_parallel data_parallel.cc data_parallel.h)

add_library(util util.h)
target_link_libraries(util absl::strings absl::time)

add_library(lambertw lambertw.cc lambertw.h)

add_library(file_passthrough file_passthrough_external.cc
  file_passthrough_external.h file_passthrough.h)
target_link_libraries(file_passthrough absl::strings)

add_library(mnist_loader mnist_loader.cc mnist_loader.h)
target_link_libraries(mnist_loader file_passthrough util)

