add_library(tempcoder tempcoder.cc tempcoder.h)
target_link_libraries(tempcoder file_passthrough util
  lambertw absl::strings absl::synchronization)

add_library(runner runner.cc runner.h)
target_link_libraries(runner data_parallel gradient_landscape_helper
  spiking_problem tempcoder util absl::strings absl::synchronization)

add_library(gradient_landscape_helper gradient_landscape_helper.cc
  gradient_landscape_helper.h)
target_link_libraries(gradient_landscape_helper data_parallel spiking_problem
  tempcoder util)

add_library(spiking_problem spiking_problem.cc spiking_problem.h)
target_link_libraries(spiking_problem file_passthrough mnist_loader tempcoder
  util absl::algorithm absl::strings)

add_executable(tempcoding_main tempcoding_main.cc)
target_link_libraries(tempcoding_main data_parallel file_passthrough
  gradient_landscape_helper runner spiking_problem tempcoder util absl::flags
  absl::flags_parse absl::memory absl::strings absl::time)
