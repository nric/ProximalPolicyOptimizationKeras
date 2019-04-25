"""This is a deterministic example of a Open Ai's proximal policy optimization actor critic algorithm PPO.
It is implemented using Tensorflow 2.0 (keras). 
Part of the code base is from https://github.com/liziniu/RL-PPO-Keras . However, the code there had errors
but mainly it did not use a GAE type reward and no entropy bonus system.
I gave my best to comment the code but I did not include a fundamental lecutre on the logic behind PPO. I highly 
recommend to watch these two videos to undestand what happens.
https://youtu.be/WxQfQW48A4A
https://youtu.be/5P7I-xPq8u8

todo: 
check noisydense
continous version

    
"""

#%%
import numpy as np
import os
import time
import gym
from collections import deque
import tensorflow as tf
import tensorflow.keras as K


#%%
class Memory:
    """class Memory: Stores experience trajectories.
    Idea is to fill the memory with trajectories (s,a,r,gae_r,s_,d) tuples and get an arbitary number of
    random batches from the memory for training. After that, clear and fill next set (on policy training).
    The memory gets filled from the Agent object.
    A special case is the batch_gae which gets calculated once when training on memory starts.
    """
    def __init__(self):
        self.batch_s = []
        self.batch_a = []
        self.batch_r = []
        self.batch_gae_r = [] #this gets set in agent make_gae which is called once on first training on memory
        self.batch_s_ = []
        self.batch_done = []
        self.GAE_CALCULATED_Q = False #make sure make_gae can only be called once
    

    def get_batch(self,batch_size):
        """simply retuns a randomized batch from the data in memory
        """
        for _ in range(batch_size):
            s,a,r,gae_r,s_,d = [],[],[],[],[],[]
            pos = np.random.randint(len(self.batch_s)) #random position
            s.append(self.batch_s[pos])
            a.append(self.batch_a[pos])
            r.append(self.batch_r[pos])
            gae_r.append(self.batch_gae_r[pos])
            s_.append(self.batch_s_[pos])
            d.append(self.batch_done[pos])
        return s,a,r,gae_r,s_,d #return randomized batches


    def store(self, s, a, s_, r, done):
        """push s,a,r,s_,done into memory (=according lists)
        """
        self.batch_s.append(s)
        self.batch_a.append(a)
        self.batch_r.append(r)
        self.batch_s_.append(s_)
        self.batch_done.append(done)


    def clear(self):
        """clear all lists (=memory)
        """
        self.batch_s.clear()
        self.batch_a.clear()
        self.batch_r.clear()
        self.batch_s_.clear()
        self.batch_done.clear()
        self.GAE_CALCULATED_Q = False


    @property
    def cnt_samples(self):
        return len(self.batch_s)


#%%
class Agent:
    def __init__(self,action_n, state_dim, training_batch_size):
        """This is the agent object.
        Main interaction is the choose_action, store transition and train_network. 
        The agent only requires the state and action spaces to fuction, other than that it is pretty general
        and should be easy to adapt for other deterministic envs.
        To understand what is happening, I recommend to look at the ppo_loss method and the build_actor method first.
        The training method itself is more or less only data preperation for calling the fit functions
        for actor and critic. But critic has a trivial loss, so all the PPO magic is in the ppo_loss function.

        """
        self.action_n = action_n
        self.state_dim = state_dim        
        #CONSTANTS
        self.TRAINING_BATCH_SIZE = training_batch_size
        self.TARGET_UPDATE_ALPHA = 0.95
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.CLIPPING_LOSS_RATIO = 0.1
        self.ENTROPY_LOSS_RATIO = 0.001
        self.TARGET_UPDATE_ALPHA = 0.9
        #create actor and critic neural networks
        self.critic_network = self._build_critic_network()
        self.actor_network = self._build_actor_network()
        #for the loss function, additionally "old" predicitons are required from before the last update.
        #therefore create another networtk. Set weights to be identical for now.
        self.actor_old_network = self._build_actor_network()
        self.actor_old_network.set_weights(self.actor_network.get_weights()) 
        #for getting an action (predict), the model requires it's ususal input, but advantage and old_prediction is only used for loss(training). So create dummys for prediction only
        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediciton = np.zeros((1, self.action_n))
        #our transition memory buffer        
        self.memory = Memory()
        

    
    def _build_actor_network(self):
        """builds and returns a compiled keras.model for the actor.
        There are 3 inputs. Only the state is for the pass though the neural net. 
        The other two inputs are exclusivly used for the custom loss function (ppo_loss).
        """
        #define inputs. Advantage and old_prediction are required to pass to the ppo_loss funktion
        state = K.layers.Input(shape=self.state_dim,name='state_input')
        advantage = K.layers.Input(shape=(1,),name='advantage_input')
        old_prediction = K.layers.Input(shape=(self.action_n,),name='old_prediction_input')
        #define hidden layers
        dense = K.layers.Dense(32,activation='relu',name='dense1')(state)
        dense = K.layers.Dense(32,activation='relu',name='dense2')(dense)
        #connect layers, output action using softmax activation
        policy = K.layers.Dense(self.action_n, activation="softmax", name="actor_output_layer")(dense)
        #make keras.Model
        actor_network = K.Model(inputs = [state,advantage,old_prediction], outputs = policy)
        #compile. Here the connection to the PPO loss fuction is made. The input placeholders are passed.
        actor_network.compile(
            optimizer='Adam',
            loss = self.ppo_loss(advantage=advantage,old_prediction=old_prediction)
            )
        #summary and return       
        actor_network.summary()
        time.sleep(1.0)
        return actor_network


    def _build_critic_network(self):
        """builds and returns a compiled keras.model for the critic.
        The critic is a simple scalar prediction on the state value(output) given an state(input)
        Loss is simply mse
        """
        #define input layer
        state = K.layers.Input(shape=self.state_dim,name='state_input')
        #define hidden layers
        dense = K.layers.Dense(32,activation='relu',name='dense1')(state)
        dense = K.layers.Dense(32,activation='relu',name='dense2')(dense)
        #connect the layers to a 1-dim output: scalar value of the state (= Q value or V(s))
        V = K.layers.Dense(1, name="actor_output_layer")(dense)
        #make keras.Model
        critic_network = K.Model(inputs=state, outputs=V)
        #compile. Here the connection to the PPO loss fuction is made. The input placeholders are passed.
        critic_network.compile(optimizer='Adam',loss = 'mean_squared_error')
        #summary and return           
        critic_network.summary()
        time.sleep(1.0)
        return critic_network
    

    def ppo_loss(self, advantage, old_prediction):
        """The PPO custom loss.
        For explanation see for example:
        https://youtu.be/WxQfQW48A4A
        https://youtu.be/5P7I-xPq8u8
        params:
            :advantage: advantage, needed to process algorithm
            :old_predictioN: prediction from "old" network, needed to process algorithm
        returns:
            :loss: keras type loss fuction (not a value but a fuction with two parameters y_true, y_pred)
        """
        #refer to Keras custom loss function intro to understand why we define a funciton inside a function.
        def loss(y_true, y_pred):
            prob = y_true * y_pred #y_true is taken action one_hot(in deterministic case) and pred is a softmax vector. prob is the probability of the taken aciton.
            old_prob = y_true * old_prediction
            ratio = prob / (old_prob + 1e-10)
            clip_ratio = K.backend.clip(ratio, min_value=1 - self.CLIPPING_LOSS_RATIO, max_value=1 + self.CLIPPING_LOSS_RATIO)
            surrogate1 = ratio * advantage
            surrogate2 = clip_ratio * advantage
            entropy_loss = (prob * K.backend.log(prob + 1e-10)) #optionally add the entropy loss to avoid getting stuck on local minima
            ppo_loss = -K.backend.mean(K.backend.minimum(surrogate1,surrogate2) + self.ENTROPY_LOSS_RATIO * entropy_loss)
            return ppo_loss
        return loss
    

    def make_gae(self):
        """Generates GAE type rewards and pushes them into memory object
        #GAE algorithm: 
            #delta = r + gamma * V(s') * mask - V(s)  |aka advantage
            #gae = delta + gamma * lambda * mask * gae |moving average smoothing
            #return(s,a) = gae + V(s)  |add value of state back to it.
        """
        gae = 0
        for i in reversed(range(self.memory.cnt_samples))):
            mask = 0 if self.memory.batch_done[i] else 1
            v = self.get_v(self.memory.batch_s[i])
            delta = self.memory.batch_r[i] + self.GAMMA * self.get_v(self.memory.batch_s_[i]) * mask - v
            gae = delta + self.GAMMA *  self.GAE_LAMBDA * mask * gae
            self.memory.batch_gae_r.append(gae+v)
        self.memory.batch_gae_r.reverse()
        self.memory.GAE_CALCULATED_Q = True


    def update_tartget_network(self):
        """Softupdate of the target network.
        In ppo, the updates of the 
        """
        alpha = self.TARGET_UPDATE_ALPHA
        actor_weights = np.array(self.actor_network.get_weights())
        actor_tartget_weights = np.array(self.actor_old_network.get_weights())
        new_weights = alpha*actor_weights + (1-alpha)*actor_tartget_weights
        self.actor_old_network.set_weights(new_weights)

    
    def choose_action(self,state):
        """chooses an action within the action space given a state.
        The action is chosen by random with the weightings accoring to the probability
        params:
            :state: np.array of the states with state_dim length
        """
        assert isinstance(state,np.ndarray)
        #reshape for predict_on_batch which requires 2d-arrays
        state = np.reshape(state,[-1,self.state_dim[0]])
        #the probability list for each action is the output of the actor network given a state
        prob = self.actor_network.predict_on_batch([state,self.dummy_advantage, self.dummy_old_prediciton]).flatten()
        #action is chosen by random with the weightings accoring to the probability
        action = np.random.choice(self.action_n,p=prob)
        return action
    

    def train_network(self):
        """Train the actor and critic networks using GAE Algorithm.
        1. Get GAE rewards
        2. reshape batches s,a,gae_r baches
        3. get value of state
        4. calc advantage
        5. get "old" precition (of target network)
        6. fit actor and critic network
        7. soft update target "old" network
        """
        #important: make gae type rewards BEFORE getting random batches if not done yet
        if not self.memory.GAE_CALCULATED_Q:
            self.make_gae()
        #get randomized mini batches
        states,actions,rewards,gae_r,next_states,dones = self.memory.get_batch(self.TRAINING_BATCH_SIZE)
       
        #create np array batches for training
        batch_s = np.vstack(states)
        batch_a = np.vstack(actions)
        batch_gae_r = np.vstack(gae_r)
        #get values of states in batch
        batch_v = self.get_v(batch_s)
        #calc advantages. required for actor loss. 
        batch_advantage = batch_gae_r - batch_v
        batch_advantage = K.utils.normalize(batch_advantage) #
        #calc old_prediction. Required for actor loss.
        batch_old_prediction = self.get_old_prediction(batch_s)
        #one-hot the actions. Actions will be the target for actor.
        batch_a_final = np.zeros(shape=(len(batch_a), self.action_n))
        batch_a_final[:, batch_a.flatten()] = 1

        #commit training
        self.actor_network.fit(x=[batch_s, batch_advantage, batch_old_prediction], y=batch_a_final, verbose=0)
        self.critic_network.fit(x=batch_s, y=batch_gae_r, epochs=1, verbose=0)
        #soft update the target network(aka actor_old). 
        self.update_tartget_network()


    def store_transition(self, s, a, s_, r, done):
        """Store the experiences transtions into memory object.
        """
        self.memory.store(s, a, s_, r, done)


    def get_v(self,state):
        """Returns the value of the state.
        Basically, just a forward pass though the critic networtk
        """
        s = np.reshape(state,(-1, self.state_dim[0]))
        v = self.critic_network.predict_on_batch(s)
        return v
    

    def get_old_prediction(self, state):
        """Makes an prediction (an action) given a state on the actor_old_network.
        This is for the train_network --> ppo_loss
        """
        state = np.reshape(state, (-1, self.state_dim[0]))
        return self.actor_old_network.predict_on_batch([state,self.dummy_advantage, self.dummy_old_prediciton])



    
    

#%%
ENV_NAME = "LunarLander-v2"
TRAIN_ITERATIONS = 1000
MAX_EPISODE_LENGTH = 1000
TRAJECTORY_BUFFER_SIZE = 32
BATCH_SIZE = 16
RENDER_EVERY = 100


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    agent = Agent(env.action_space.n,env.observation_space.shape,BATCH_SIZE)
    samples_filled = 0

    for cnt_episode in range(TRAIN_ITERATIONS):
        s = env.reset()
        r_sum = 0
        for cnt_step in range(MAX_EPISODE_LENGTH):
            #sometimes render
            if cnt_episode % RENDER_EVERY == 0 :
                env.render()
            #get action from agent given state
            a = agent.choose_action(s)
            #get s_,r,done
            s_, r, done, _ = env.step(a)
            r /= 100
            r_sum += r
            if done:
                r = -1
            #store transitions to agent.memory
            agent.store_transition(s, a, s_, r, done)
            samples_filled += 1
            #train in batches one buffer is filled with samples.
            if samples_filled % TRAJECTORY_BUFFER_SIZE == 0 and samples_filled != 0:
                #To be sample efficient, sample as often as statistically necearry to 
                # use all availible samples in memory. Imortant to sample randomly 
                # to keep the training data independant and identically distributed IID
                for _ in range(TRAJECTORY_BUFFER_SIZE // BATCH_SIZE):
                    agent.train_network()
                agent.memory.clear()
                samples_filled = 0
            #set state to next_state
            s = s_
            if done:
                break
        if cnt_episode % 10 == 0:
            print(f"Episode:{cnt_episode}, step:{cnt_step}, r_sum:{r_sum}")


