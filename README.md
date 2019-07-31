# Ihmehimmeli

This repository contains code for project Ihmehimmeli.

The objective of Ihmehimmeli is to build recurrent architectures for
state-based spiking neural networks that encode information in the timing of
individual neuron spikes. Spike-based temporal coding allows a natural and
energy-efficient solution for the encoding and processing of real-world analog
signals. This approach can potentially evolve into native interfaces between
artficial and biological neural networks. Similar to the way that biological
brains have evolved to use temporal coding for the rapid processing of sensory
information, we expect that equivalent develpments in spiking networks will be
a key future step in the advancement of general artificial intelligence.

Our model is described in detail in this paper.
***TODO***: Add paper link when it's released.

## Build instructions

Compiling this project requires CMake and a C++11 compliant compiler.
It was tested with CMake 3.12.1 and g++ 7.3.0, though it will likely work with
other versions and compilers.

``` shell
git clone https://github.com/google/ihmehimmeli
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

### Train a spiking network on MNIST.
``` shell
cd build
tempcoding/tempcoding_main -problem=mnist -n_train=54000 -n_validation=6000 -n_test=10000  -batch_size=32 -clip_derivative=539.69973904211679 -decay_rate=0.18176949150701854 -fire_threshold=1.1673205005788956 -learning_rate=0.0010186407877494507 -learning_rate_pulses=0.09537534860701444 -n_hidden=340 -n_pulses=10 -nonpulse_weight_mean_multiplier=-0.2754188425913906 -penalty_no_spike=48.374830659132556 -pulse_weight_mean_multiplier=7.8391245503824578 -update_all_datapoints=true -use_adam=true -n_epochs=100 -mnist_data_path=../data/mnist
```

### Test a spiking network on MNIST
Two networks reported in the paper are available under `tempcoding/networks/`:
a `slow_network` that achieves the best accuracy but is slow, and
a `fast_network` that is less accurate but makes decisions very fast.

``` shell
cd build
tempcoding/tempcoding_main -model_to_test=tempcoding/networks/slow_network -problem=mnist -n_test=10000 -n_train=60000 -n_validation=0 -decay_rate=0.181769 -mnist_data_path=../data/mnist
```

## License

Apache 2.0; see [LICENSE](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by Google
and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
