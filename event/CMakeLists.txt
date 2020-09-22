add_library(event
  event.cc
  event.h
  loss.cc
  loss.h
  potential.cc
  potential.h
  problem.cc
  problem.h
  adam.h
  runner.cc
  runner.h
  spike_queue.cc
  spike_queue.h
)

target_link_libraries(event
  absl::strings
  absl::str_format
  lambertw
  mnist_loader
)

add_executable(mnist mnist_main.cc)
target_link_libraries(mnist event data_parallel)
