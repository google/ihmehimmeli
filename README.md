# Ihmehimmeli

This repository contains code for project Ihmehimmeli. The model is described in the paper:

I.M. Comsa, K. Potempa, L. Versari, T. Fischbacher, A. Gesmundo, J. Alakuijala (2019). “[Temporal coding in spiking neural networks with alpha synaptic function](https://arxiv.org/abs/1907.13223)”, arXiv:1907.13223, July 2019.

The objective of Ihmehimmeli is to build recurrent architectures for
state-based spiking neural networks that encode information in the timing of
individual neuron spikes. Spike-based temporal coding allows a natural and
energy-efficient solution for the encoding and processing of real-world analog
signals. This approach can potentially evolve into native interfaces between
artficial and biological neural networks. Similar to the way that biological
brains have evolved to use temporal coding for the rapid processing of sensory
information, we expect that equivalent develpments in spiking networks will be
a key future step in the advancement of general artificial intelligence.

## Spiking autoencoder
This branch provides an implementation of a spiking autoencoder that encodes
information with temporal coding and is trained using backpropagation.

## Build instructions

Compiling this project requires CMake and a C++11 compliant compiler.
It was tested with CMake 3.12.1 and g++ 7.3.0, though it will likely work with
other versions and compilers.

``` shell
git clone --branch autoencoder https://github.com/google/ihmehimmeli
cd ihmehimmeli
git clone https://github.com/abseil/abseil-cpp.git
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 12
```

## Train a spiking network on MNIST

### Download the MNIST dataset
``` shell
./download_mnist.sh
```

### Train a spiking autoencoder on MNIST.
``` shell
cd build
tempcoding/tempcoding_main -problem=mnist_ae -print_test_set=false -print_test_stats=true -n_train=54000 -n_validation=6000 -n_epochs=100 -n_test=10000 -noisy_targets_ae=false -n_runs=1 -wait_for_input_between_runs=false -noise_factor_ae=0.0 -latency_ae=1 -batch_size=7 -clip_derivative=386.058227 -decay_rate=0.172556 -fire_threshold=0.834236 -learning_rate=0.001676 -learning_rate_pulses=0.001441 -n_hidden=32 -n_pulses=10 -nonpulse_weight_mean_multiplier=-7.038462 -penalty_no_spike=6.250341 -pulse_weight_mean_multiplier=4.213218 -mnist_data_path=../data/mnist
```

## License

Apache 2.0; see [LICENSE](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by Google
and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
