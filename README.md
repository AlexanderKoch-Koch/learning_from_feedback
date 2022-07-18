# feedback_learning
## Learning to predict bit sequence with and without binary error feedback
experiments/train_single_step_dreamer.py was used to run this experiment.
The number of runs with different random seeds, the log directory and whether to use 
feedback observations can be changed in this file.
The experiments/logs/make_sequence_guessing_figure.py script was used to create
the training progress figures in pdf format. It produces one pdf file for every sequence length range.
So one is the success rate on guessing sequences of length between 1 and 15 bits and so on.

# Open-Loop Dreamer
experiments/train_open_loop_dreamer contains the script that trains the open-loop Dreamer algororithm on the Pendulum-v0 environment.
The results can be visualized with the experiments/make_open_loop_dreamer_figure.py script.

# Simple Feedback Reacher
This part is not yet finished. The environment code is in learning_from_feedback/envs/simple_feedback_reacher.
There is one training script that trains a simplified dreamer algorithm on this environment in experiments/simple_feedback_reacher/train_single_step_dreamer.py.
Another training script in experiments/simple_feedback_reacher/train_state_dreamer.py trains a more complete Dreamer agent.

# Literature List
## Hierarchical RL
[A framework for temporal abstraction in reinforcement learning](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)
(One of the first papers in Hierarchical RL) \
[Feudal Networks for Hierarchical Reinforcement Learning](http://proceedings.mlr.press/v70/vezhnevets17a.html) \
[Hierarchical Skills for Efficient Exploration](https://arxiv.org/abs/2110.10809) \
[Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation](https://arxiv.org/abs/1604.06057)

## Model based RL
[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
(Introduces the Dreamer algorithm)\
[Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)
(Improved Dreamer algorithm for discrete control)\
[Learning Continuous Control Policies by Stochastic Value Gradients](https://arxiv.org/abs/1510.09142) \
[End-to-End Differentiable Physics for Learning and Control](http://papers.neurips.cc/paper/7948-end-to-end-differentiable-physics-for-learning-and-control.pdf) \
[Making the World Differentiable: On Using Self-Supervised Fully Recurrent Neural Networks for Dynamic Reinforcement Learning and Planning in Non-Stationary Environments ](https://people.idsia.ch/~juergen/FKI-126-90_(revised)bw_ocr.pdf)
(Old paper by Schmidhuber on using learned enviroment models to calculate policy gradients)


# Future Work
## Improve performance on the simple_feedback_reacher environment
The environment model still takes a very long time to train.
Maybe it is necessary to use a specialized network architecture instead of just fully connected layers.

## Implement more general robotics environment with natural language instructions and feedback
The instruction might be "pick up the bottle".
In case of an unsuccessfull attempt the feedback could be "The bottle is the green object on the right".
The idea is that it is easier for the agent to learn what a green object is than to learn what a bottle is.


## Develop hierarchical RL algorithm
The Dreamer algorithm has to predict at least T steps into the future when rewards are delayed by up to T steps.
A hierarchical RL algorithm might be able to improve a policy in a fraction of T steps.