# ProximalPolicyOptimizationKeras
This is a deterministic Tensorflow 2.0 (keras) implementation of a Open Ai's proximal policy optimization actor critic algorithm PPO.

This is a deterministic Tensorflow 2.0 (keras) implementation of a Open Ai's proximal policy optimization actor critic algorithm PPO.

Here, it is tested with open ai gym lunar lander v2 but should be easy to adapt to any env that takes an action with a
defind action space size and returns reward,next_state,done each step.

Good part of the code base is from https://github.com/liziniu/RL-PPO-Keras . However, the code there had errors
but mainly it did not use a GAE type reward and no entropy bonus system and was using non IID data (not randomized baches).

I gave my best to comment the code but I did not include a fundamental lecutre on the logic behind PPO. I highly 
recommend to watch these two videos to undestand what happens.
https://youtu.be/WxQfQW48A4A
https://youtu.be/5P7I-xPq8u8
