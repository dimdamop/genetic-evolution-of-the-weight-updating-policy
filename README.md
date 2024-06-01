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
   affects their chances of getting mutated to the next round. Additionally, the top 25 % of the
   participants are copied without any mutation.
8. The forward passes of the individual zero (ie., the first model all subsequent mutations inherit
   from) are functionally equivalent to those of a standard deep learning (DL) model.


# Recent bibliography

 * https://arxiv.org/abs/1810.01222
 * https://arxiv.org/abs/1712.06567
 * https://arxiv.org/abs/1802.06070
 * https://www.sciencedirect.com/science/article/abs/pii/B9780323961042000026

# Legal

A substantial portion of the Python code included in this repository is a slightly modified copy of
that of `jumanji.training` of InstaDeep Ltd. The modules that are such copies list the corresponding
modifications that took place at the end of their header.
