# feedback_learning
## Learning to predict bit sequence with and without binary error feedback
![fadfas](Sequence_guessing_model.png)
experiments/train_single_step_dreamer.py was used to run this experiment.
The number of runs with different random seeds, the log directory and whether to use 
feedback observations can be changed in this file.
The experiments/logs/make_sequence_guessing_figure.py script was used to create
the training progress figures in pdf format. It produces one pdf file for every sequence length range.
So one is the success rate on guessing sequences of length between 1 and 15 bits and so on.


# Literature List
## Hierarchical RL
[A framework for temporal abstraction in reinforcement learning](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)

## Model based RL
[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
