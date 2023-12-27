# Summary

What if the update of the weights of the network is not proportional to the respective gradients?

This repository explores this question in the context of reinforcement learning (RL). The main
ideas are the following:

1. We use vanilla RL learning frameworks (PPO, Q-Learning, etc).
2. We conduct tournaments of competing learnt models, each lasting a predefined number of epochs.
3. Each learnt model has its own rules for the forward passes (inference and gradient computation)
   of its neurons.
5. These model-specific rules are mutations of those of the winners of the previous tournament.
6. The genome on which these mutations take place is able to describe a subset of the Python
   language that covers the standard forward passes found in contemporary neuron models.
7. The forward passes of the first individual (ie, the first model that all subsequent mutations
   inherit from) are functionally equivalent to those of a standard deep learning (DL) model.
