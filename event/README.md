# Event-based spiking network model

This model closely follows the model in `tempcoding/`, but allows for more
flexibility:

- Network structure is arbitrary and, in particular, can be non-acyclic.
  Synchronization pulses can be connected to any subset of neurons.
- Neurons can spike more than once. An network parameter `refractory_period`
  controls the amount of time for which a neuron ignores incoming spikes after
  spiking.
- Each spike still produces a time-varying potential, according to some
  potential function (so far only the alpha function is implemented).
- The network is still trained with backpropagation, applying a penalty to
  connections on which a spike was sent that did not result in a spike from the
  receiving neuron.
