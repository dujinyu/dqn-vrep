# -*- coding: utf-8 -*-
"""
Deep Q-network implementation with chainer and rlglue.
"""
import copy
#import json
#import datetime
import pickle
import math
import numpy as np

import chainer
from chainer import cuda, FunctionSet, Variable, optimizers
import chainer.functions as F

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.utils import TaskSpecVRLGLUE3

#from operator import itemgetter
#from vrepUtils.pidcontrol import Hover_PID_Controller

class DQN_class:
    # Hyper-Parameters
    gamma = 0.99                       # Discount factor
    initial_exploration = 5*10**4      # 10**4  # Initial exploratoin. original: 5x10^4
    replay_size = 32                   # Replay (batch) size
    target_model_update_freq = 10**4   # Target update frequancy. original: 10^4
    data_size = 10**6                  # Data size of history. original: 10^6
    num_of_actions = 2                 # Action dimention
    num_of_states = 12                 # State dimention
    
    def __init__(self):
                  
        print "Initializing DQN..."
#	Initialization of Chainer 1.1.0 or older.
#        print "CUDA init"
#        cuda.init()

        print "Model Building"
#        self.model = FunctionSet(
#            l1=F.Convolution2D(4, 32, ksize=8, stride=4, nobias=False, wscale=np.sqrt(2)),
#            l2=F.Convolution2D(32, 64, ksize=4, stride=2, nobias=False, wscale=np.sqrt(2)),
#            l3=F.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, wscale=np.sqrt(2)),
#            l4=F.Linear(3136, 512, wscale=np.sqrt(2)),
#            q_value=F.Linear(512, self.num_of_actions,
#                             initialW=np.zeros((self.num_of_actions, 512),
#                                               dtype=np.float32))
#        ).to_gpu()
        
#        self.critic = FunctionSet(
#            l1=F.Linear(self.num_of_actions+self.num_of_states,512),
#            l2=F.Linear(512,256),
#            l3=F.Linear(256,128),
#            q_value=F.Linear(128,1,initialW=np.zeros((1,128),dtype=np.float32))
#        ).to_gpu()
#        
#        self.actor = FunctionSet(
#            l1=F.Linear(self.num_of_states,512),
#            l2=F.Linear(512,256),
#            l3=F.Linear(256,128),
#            a_value=F.Linear(128,self.num_of_actions,initialW=np.zeros((1,128),dtype=np.float32))
#        ).to_gpu()
        
        self.critic = FunctionSet(
            l1=F.Linear(self.num_of_actions+self.num_of_states,1024),
            l2=F.Linear(1024,512),
            l3=F.Linear(512,256),
            l4=F.Linear(256,128),
            q_value=F.Linear(128,1,initialW=np.zeros((1,128),dtype=np.float32))
        ).to_gpu()
        
        self.actor = FunctionSet(
            l1=F.Linear(self.num_of_states,1024),
            l2=F.Linear(1024,512),
            l3=F.Linear(512,256),
            l4=F.Linear(256,128),
            a_value=F.Linear(128,self.num_of_actions,initialW=np.zeros((1,128),dtype=np.float32))
        ).to_gpu()
        
#        self.critic = FunctionSet(
#            l1=F.Linear(self.num_of_actions+self.num_of_states,1024,wscale=0.01*math.sqrt(self.num_of_actions+self.num_of_states)),
#            l2=F.Linear(1024,512,wscale=0.01*math.sqrt(1024)),
#            l3=F.Linear(512,256,wscale=0.01*math.sqrt(512)),
#            l4=F.Linear(256,128,wscale=0.01*math.sqrt(256)),
#            q_value=F.Linear(128,1,wscale=0.01*math.sqrt(128))
#        ).to_gpu()
#        
#        self.actor = FunctionSet(
#            l1=F.Linear(self.num_of_states,1024,wscale=0.01*math.sqrt(self.num_of_states)),
#            l2=F.Linear(1024,512,wscale=0.01*math.sqrt(1024)),
#            l3=F.Linear(512,256,wscale=0.01*math.sqrt(512)),
#            l4=F.Linear(256,128,wscale=0.01*math.sqrt(256)),
#            a_value=F.Linear(128,self.num_of_actions,wscale=0.01*math.sqrt(128))
#        ).to_gpu()
        
        self.critic_target = copy.deepcopy(self.critic) 
        self.actor_target = copy.deepcopy(self.actor)
        
        print "Initizlizing Optimizer"
        #self.optim_critic = optimizers.RMSpropGraves(lr=0.0001, alpha=0.95, momentum=0.95, eps=0.0001)
        #self.optim_actor = optimizers.RMSpropGraves(lr=0.0001, alpha=0.95, momentum=0.95, eps=0.0001)
        self.optim_critic = optimizers.Adam(alpha=0.0001)
        self.optim_actor = optimizers.Adam(alpha=0.0001)
        self.optim_critic.setup(self.critic)
        self.optim_actor.setup(self.actor)
        
#        self.optim_critic.add_hook(chainer.optimizer.WeightDecay(0.00001))
#        self.optim_critic.add_hook(chainer.optimizer.GradientClipping(10))
#        self.optim_actor.add_hook(chainer.optimizer.WeightDecay(0.00001))
#        self.optim_actor.add_hook(chainer.optimizer.GradientClipping(10))

        # History Data :  D=[s, a, r, s_dash, end_episode_flag]
        self.D = [np.zeros((self.data_size, self.num_of_states), dtype=np.float32),
                  np.zeros((self.data_size, self.num_of_actions), dtype=np.float32),
                  np.zeros((self.data_size, 1), dtype=np.float32),
                  np.zeros((self.data_size, self.num_of_states), dtype=np.float32),
                  np.zeros((self.data_size, 1), dtype=np.bool)]
                  
#        with open('dqn_dump.json', 'a') as f:
#            json.dump(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), f)
#            f.write('\n')
#            json.dump({"alpha": 0.00001, "beta1": 0.7, "beta2": 0.999, "weight_decay": 0.00001}, f)
#            f.write('\n')
#            f.close()
        #self.x_PID = Hover_PID_Controller(12.1, 1.25)
        #self.y_PID = Hover_PID_Controller(12.1, 1.25)

    def forward(self, state, action, Reward, state_dash, episode_end):
        num_of_batch = state.shape[0]
        
        s = Variable(cuda.to_gpu(np.concatenate([state, action],1)))
        s_dash = Variable(cuda.to_gpu(state_dash))

        Q = self.Q_func(s)  # Get Q-value
        
        # Generate Target through target nets
        action_dash_tmp = self.A_func_target(s_dash) 
        action_dash = np.asanyarray(action_dash_tmp.data.get(), dtype=np.float32)
        tmp_dash = Variable(cuda.to_gpu(np.concatenate([state_dash, action_dash],1)))
        Q_dash_tmp = self.Q_func_target(tmp_dash)
        Q_dash = np.asanyarray(Q_dash_tmp.data.get(), dtype=np.float32)       
        target = np.asanyarray(Q.data.get(), dtype=np.float32)

        for i in xrange(num_of_batch):
            if not episode_end[i][0]:
                tmp_ = Reward[i] + self.gamma * Q_dash[i]
            else:
                tmp_ = Reward[i]

            target[i] = tmp_

        # TD-error clipping
        td = Variable(cuda.to_gpu(target)) - Q  # TD error
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        zero_val = Variable(cuda.to_gpu(np.zeros((self.replay_size, 1), dtype=np.float32)))
        loss = F.mean_squared_error(td_clip, zero_val)
        
        return loss, Q

    def updateActor(self, state):
        num_of_batch = state.shape[0]
        A_max = 1.0
        A_min = -1.0
        
        A = self.A_func(Variable(cuda.to_gpu(state)))
        tmp = Variable(cuda.to_gpu(np.concatenate([state, A.data.get()],1)))
        Q = self.Q_func(tmp)
        
        # Backward prop towards actor net
        #self.critic.zerograds()
        #self.actor.zerograds()
        Q.grad = cuda.to_gpu(np.ones((num_of_batch, 1), dtype=np.float32)*(-1.0))
#        Q.grad = Q.data*(-1.0)
        Q.backward()
        A.grad = tmp.grad[:,-self.num_of_actions:]
        print("sample_A.grad: "+str(A.grad[0]))
        for i in xrange(num_of_batch):
            for j in xrange(self.num_of_actions):
                if A.grad[i][j] < 0:
                    A.grad[i][j] *= (A_max-A.data[i][j])/(A_max-A_min)
                elif A.grad[i][j] > 0:
                    A.grad[i][j] *= (A.data[i][j]-A_min)/(A_max-A_min)
            
        A.backward()
        self.optim_actor.update()
        print("sample_A.grad: "+str(A.grad[0]))
        
    def stockExperience(self, time,
                        state, action, reward, state_dash,
                        episode_end_flag):
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.D[0][data_index] = state
            self.D[1][data_index] = action
            self.D[2][data_index] = reward
        else:
            self.D[0][data_index] = state
            self.D[1][data_index] = action
            self.D[2][data_index] = reward
            self.D[3][data_index] = state_dash
        self.D[4][data_index] = episode_end_flag

    def experienceReplay(self, time):

        if self.initial_exploration < time:
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))
                #reward_list = list(self.D[2])
                #replay_index = [i[0] for i in sorted(enumerate(reward_list),key=itemgetter(1),reverse=True)[:32]]
                #replay_index = np.asarray(replay_index).reshape(32,1)
                
            s_replay = np.ndarray(shape=(self.replay_size, self.num_of_states), dtype=np.float32)
            a_replay = np.ndarray(shape=(self.replay_size, self.num_of_actions), dtype=np.float32)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(self.replay_size, self.num_of_states), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
            for i in xrange(self.replay_size):
                s_replay[i] = np.asarray(self.D[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = np.asarray(self.D[1][replay_index[i]], dtype=np.float32)
                r_replay[i] = self.D[2][replay_index[i]]
                s_dash_replay[i] = np.asarray(self.D[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.D[4][replay_index[i]]

            #s_replay = cuda.to_gpu(s_replay)
            #s_dash_replay = cuda.to_gpu(s_dash_replay)

            # Gradient-based critic update
            self.optim_critic.zero_grads()
            loss, q = self.forward(s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay)
            loss.backward()
            self.optim_critic.update()
            
            # Update the actor
            self.optim_critic.zero_grads()
            self.optim_actor.zero_grads()
            self.updateActor(s_replay)
            
            self.soft_target_model_update()
            
            print "AVG_Q %f" %(np.average(q.data.get()))
            print("loss " + str(loss.data))
            
#            with open('dqn_dump.json', 'a') as f:
#                json.dump({"time": time, "avg_Q": float(np.average(q.data.get())), "loss": float(loss.data)}, f)
#                f.write('\n')
#                f.close()

    def Q_func(self, state):
#        h1 = F.relu(self.critic.l1(state))
#        h2 = F.relu(self.critic.l2(h1))
#        h3 = F.relu(self.critic.l3(h2))
#        Q = self.critic.q_value(h3)
        h1 = F.relu(self.critic.l1(state))
        h2 = F.relu(self.critic.l2(h1))
        h3 = F.relu(self.critic.l3(h2))
        h4 = F.relu(self.critic.l4(h3))
        Q = self.critic.q_value(h4)
        return Q

    def Q_func_target(self, state):
#        h1 = F.relu(self.critic_target.l1(state))
#        h2 = F.relu(self.critic_target.l2(h1))
#        h3 = F.relu(self.critic.l3(h2))
#        Q = self.critic_target.q_value(h3)   
        h1 = F.relu(self.critic_target.l1(state))
        h2 = F.relu(self.critic_target.l2(h1))
        h3 = F.relu(self.critic_target.l3(h2))
        h4 = F.relu(self.critic.l4(h3))
        Q = self.critic_target.q_value(h4)
        return Q
        
    def A_func(self, state):
#        h1 = F.relu(self.actor.l1(state))
#        h2 = F.relu(self.actor.l2(h1))
#        h3 = F.relu(self.actor.l3(h2))
#        A = self.actor.a_value(h3)
        h1 = F.relu(self.actor.l1(state))
        h2 = F.relu(self.actor.l2(h1))
        h3 = F.relu(self.actor.l3(h2))
        h4 = F.relu(self.actor.l4(h3))
        A = self.actor.a_value(h4)
        return A

    def A_func_target(self, state):
#        h1 = F.relu(self.actor_target.l1(state))
#        h2 = F.relu(self.actor_target.l2(h1))
#        h3 = F.relu(self.actor.l3(h2))
#        A = self.actor_target.a_value(h3)
        h1 = F.relu(self.actor_target.l1(state))
        h2 = F.relu(self.actor_target.l2(h1))
        h3 = F.relu(self.actor_target.l3(h2))
        h4 = F.relu(self.actor.l4(h3))
        A = self.actor_target.a_value(h4)
        return A

    def e_greedy(self, state, epsilon):
        s = Variable(state)
        A = self.A_func(s)
        A = A.data
        if np.random.rand() < epsilon:
            action = np.random.uniform(-1.,1.,(1,self.num_of_actions)).astype(np.float32)
#            action = np.zeros((1,self.num_of_actions),dtype=np.float32)
#            if state[0,0] > 0:
#                action[0,0] = np.random.uniform(0.0,0.5)
#            elif state[0,0] < 0:
#                action[0,0] = np.random.uniform(-0.5,0.0)                
#            if state[0,1] < 0:            
#                action[0,1] = np.random.uniform(0.0,0.5)
#            elif state[0,1] > 0:
#                action[0,1] = np.random.uniform(-0.5,0.0)
            #print("teststate"+str(state))
            #action[0,0] = -self.x_PID.getCorrection(state[0][0], 0.0)
            #action[0,1] = self.y_PID.getCorrection(state[0][1], 0.0)
            print "RANDOM"
        else:
            action = A.get()
            print "GREEDY"
            #print(str(action))
        return action

    def hard_target_model_update(self):
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

    def soft_target_model_update(self, tau=0.001):
        self.critic_target.l1.W.data = tau*self.critic.l1.W.data + (1-tau)*self.critic_target.l1.W.data
        self.critic_target.l2.W.data = tau*self.critic.l2.W.data + (1-tau)*self.critic_target.l2.W.data
        self.critic_target.l3.W.data = tau*self.critic.l3.W.data + (1-tau)*self.critic_target.l3.W.data
        self.critic_target.l4.W.data = tau*self.critic.l4.W.data + (1-tau)*self.critic_target.l4.W.data
        self.critic_target.q_value.W.data = tau*self.critic.q_value.W.data + (1-tau)*self.critic_target.q_value.W.data
        self.actor_target.l1.W.data = tau*self.actor.l1.W.data + (1-tau)*self.actor_target.l1.W.data
        self.actor_target.l2.W.data = tau*self.actor.l2.W.data + (1-tau)*self.actor_target.l2.W.data
        self.actor_target.l3.W.data = tau*self.actor.l3.W.data + (1-tau)*self.actor_target.l3.W.data
        self.actor_target.l4.W.data = tau*self.actor.l4.W.data + (1-tau)*self.actor_target.l4.W.data
        self.actor_target.a_value.W.data = tau*self.actor.a_value.W.data + (1-tau)*self.actor_target.a_value.W.data

class dqn_agent(Agent):  # RL-glue Process
    lastAction = Action()
    policyFrozen = False

    def agent_init(self, taskSpec):
        
        # taskspec check
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpec)
        if TaskSpec.valid:
            assert len(TaskSpec.getDoubleObservations())>0, "expecting at least one continuous observation"
            self.state_range = np.asarray(TaskSpec.getDoubleObservations())
            
            # Check action form, and then set number of actions
            assert len(TaskSpec.getIntActions())==0, "expecting no discrete actions"
            assert len(TaskSpec.getDoubleActions())==2, "expecting 1-dimensional continuous actions"

        else:
            print "Task Spec could not be parsed"
            
        self.lbounds=[]
        self.ubounds=[]
        
        for r in self.state_range:
            self.lbounds.append(r[0])
            self.ubounds.append(r[1])
            
        self.lbounds = np.array(self.lbounds)
        self.ubounds = np.array(self.ubounds)
        
        # Some initializations for rlglue
        self.lastAction = Action()

        self.time = 0
        self.epsilon = 1.0  # Initial exploratoin rate

        # Pick a DQN from DQN_class
        self.DQN = DQN_class()  
        
    def agent_start(self, observation):

        # Observation
        obs_array = np.array(observation.doubleArray)

        # Initialize State
        #self.state = self.rescale_value(obs_array)
        self.state = obs_array
        #print("state1:"+str(self.state))
        state_ = cuda.to_gpu(np.asanyarray(self.state.reshape(1,12), dtype=np.float32))
        
        # Generate an Action e-greedy
        returnAction = Action()
        action = self.DQN.e_greedy(state_, self.epsilon)
        #print(str(action))
        returnAction.doubleArray = action[0].tolist()

        # Update for next step
        self.lastAction = copy.deepcopy(returnAction)
        self.last_state = self.state.copy()
        self.last_observation = obs_array

        return returnAction

    def agent_step(self, reward, observation):

        # Observation
        obs_array = np.array(observation.doubleArray)
        #print "state: %3f %3f %3f %3f" % (obs_array[0],obs_array[1],obs_array[2],obs_array[3])
        # Compose State : 4-step sequential observation
        #self.state = self.rescale_value(obs_array)
        self.state = obs_array
        #print("state2:"+str(self.state))
        #print "state: %3f %3f %3f %3f" % (self.state[0],self.state[1],self.state[2],self.state[3])
        state_ = cuda.to_gpu(np.asanyarray(self.state.reshape(1,12), dtype=np.float32))
        #print("state2_:"+str(state_))
        # Exploration decays along the time sequence
        if self.policyFrozen is False:  # Learning ON/OFF
            if self.DQN.initial_exploration < self.time:
                self.epsilon -= 1.0/10**6
                if self.epsilon < 0.1:
                    self.epsilon = 0.1
                eps = self.epsilon
            else:  # Initial Exploation Phase
                print "Initial Exploration : %d / %d steps" % (self.time, self.DQN.initial_exploration)
                eps = 1.0
        else:  # Evaluation
                print "Policy is Frozen"
                eps = 0.05

        # Generate an Action by e-greedy action selection
        returnAction = Action()
        action = self.DQN.e_greedy(state_, eps)
        #print(str(action))
        returnAction.doubleArray = action[0].tolist()

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.DQN.stockExperience(self.time,
                                     self.last_state,
                                     np.asarray(self.lastAction.doubleArray,dtype=np.float32),
                                     reward,
                                     self.state, False)
            self.DQN.experienceReplay(self.time)

        # Target model update
#        if self.DQN.initial_exploration < self.time and np.mod(self.time, self.DQN.target_model_update_freq) == 0:
#            print "########### MODEL UPDATED ######################"
#            self.DQN.hard_target_model_update()

        # Simple text based visualization
        print 'Time Step %d / ACTION  %s / REWARD %.5f / EPSILON  %.5f' % (self.time,str(action[0]),reward,eps)

        # Updates for next step
        self.last_observation = obs_array

        if self.policyFrozen is False:
            self.lastAction = copy.deepcopy(returnAction)
            self.last_state = self.state.copy()
            self.time += 1

        return returnAction

    def agent_end(self, reward):  # Episode Terminated

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.DQN.stockExperience(self.time,
                                     self.last_state,
                                     np.asarray(self.lastAction.doubleArray,dtype=np.float32),
                                     reward, 
                                     self.last_state, True)
            self.DQN.experienceReplay(self.time)

        # Target model update
#        if self.DQN.initial_exploration < self.time and np.mod(self.time, self.DQN.target_model_update_freq) == 0:
#            print "########### MODEL UPDATED ######################"
#            self.DQN.hard_target_model_update()

        # Simple text based visualization
        print ' REWARD %.5f  / EPSILON  %.5f' % (reward, self.epsilon)

        # Time count
        if self.policyFrozen is False:
            self.time += 1

    def agent_cleanup(self):
        pass

    def agent_message(self, inMessage):
            
        if inMessage.startswith("freeze learning"):
            self.policyFrozen = True
            return "message understood, policy frozen"

        if inMessage.startswith("unfreeze learning"):
            self.policyFrozen = False
            return "message understood, policy unfrozen"

        if inMessage.startswith("save model"):
            with open('dqn_critic.dat', 'w') as f:
                pickle.dump(self.DQN.critic, f)
            with open('dqn_actor.dat', 'w') as f:
                pickle.dump(self.DQN.actor, f)
            return "message understood, model saved"
            
    def rescale_value(self, state,to_max=1.0,to_min=0.0):
        return self.scale_value(state,self.lbounds,self.ubounds,to_min,to_max)
        
    def scale_value(self,s,from_a,from_b,to_a,to_b):
        return (to_a) + (((np.array(s)-from_a)/(from_b-from_a))*((to_b)-(to_a)))
        
if __name__ == "__main__":
    AgentLoader.loadAgent(dqn_agent())
