# Summary

What if the update of the weights of the network is not based on gradients?

This repository explores this question in the context of reinforcement learning (RL). The main
ideas are the following:

1. We use vanilla RL learning frameworks (PPO, Q-Learning, etc).
2. We conduct rounds of competing predictors, each allocated a predefined time budget for training.
3. The predictors differ from each both in terms of their forward and backward passes (inference and
   gradient computation) and in terms of their weight initialization.
5. The forward and backward passes are mutations of some randomly chosen participant of the previous
   round and the weights are initialized by those of that previous participant.
6. These mutations are based on a grammar that covers a subset of the Python language.
7. The ranking of the participants of the previous round in terms of performance in the testing set
   affects their chances of getting mutated to the next round. Additionally, the top 50 % of the
   participants are copied without any mutation.
8. The forward passes of the individual zero (ie., the first model all subsequent mutations inherit
   from) are functionally equivalent to those of a standard deep learning (DL) model.
