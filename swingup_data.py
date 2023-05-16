import gym
import gym_cartpole_swingup
import numpy as np
import random
import torch
import copy
import math
#a code which collect episode sample and calculate v function
env = gym.make("CartPoleSwingUp-v1")
class collect_data(object):
    def __init__(self, trajectories, decay_for_v, seed):
        self.tra = trajectories
        self.decay = decay_for_v
        self.seed = seed

    def letsgobrandon1(self, weights, nn, index, var):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        R = []
        #training_data for states
        training_data = []
        actions = []
        mean = []
        variance = []
        while not training_data:  # == []:
            for i in range(self.tra):
                done = False
                pre_obs = []
                #reward for current episode of each time step
                reward = []
                game_memo = []
                env.reset()
                while not done:
                    if len(pre_obs)!=0:
                        #act goes from -1 to 1
                        m = nn.forward(pre_obs,weights,index)
                        act = torch.normal(mean=float(m), std=nn.var+var)#current state, current action
                        if act >= 1:
                            act = torch.tensor(1)
                        elif act <=-1:
                            act = torch.tensor(-1)
                        game_memo.append([pre_obs, act, m, float(nn.var+var)])
                        act = np.array(act)
                    else:
                        #at initial state of [0 0 cospi sinpi 0], sample an action
                        act = env.action_space.sample()
                    #transition
                    obs, rew, done, info = env.step(act)
                    obs = torch.tensor(obs,dtype=torch.float32).reshape([1,5])
                    if len(pre_obs)!=0:
                        reward.append(rew)
                    pre_obs = obs
                    #env.render()
                #reward of each episode
                R.append(reward)
                training_data.append([i[0] for i in game_memo])
                actions.append([i[1] for i in game_memo])
                mean.append([i[2] for i in game_memo])
                variance.append([i[3] for i in game_memo])
        return training_data,R,actions,mean,variance
    def train_data_v(self,training_data,R):
        #calculate value function for each step in each episode
        v_func = []
        tem = np.ones([499,1])
        for i in range(499):
            tem[i] = self.decay**i
        for epi in range(len(training_data)):
            epi_rew = (np.array(R[epi],dtype=object)).reshape([len(R[epi]),1])
            epi_tem = tem[:len(R[epi])]
            for step in range(len(training_data[epi])):
                s = training_data[epi][step]
                re = epi_rew[step:]
                step_tem = epi_tem[:len(R[epi])-step]
                v = torch.tensor(float(step_tem.T.dot(re)))
                v_func.append([s,v])
        return v_func
    def best_episodes(self,training_data,R,actions,mean,variance,part):
        #pick up episodes with best reward
        sum_each_epi = [sum(a) for a in R]
        t = copy.deepcopy(sum_each_epi)
        max_training_data = []
        max_R = []
        max_actions = []
        max_mean = []
        max_variance = []
        max_index = []
        for _ in range(part):
            number = max(t)#best reward
            index = t.index(number)#corres. index
            t[index] = 0
            max_training_data.append(training_data[index])
            max_R.append(R[index])
            max_actions.append(actions[index])
            max_mean.append(mean[index])
            max_variance.append(variance[index])
            max_index.append(index)
        t = []
        return max_training_data,max_R,max_actions,max_mean,max_variance
    def save(self,R,tra,avr_reward,best,worst,m,upper,lower,avr_var,variance):
        #avr_reward.append(sum([sum(a) for a in R])/len(R))#avr over each epoch
        best.append(max([sum(a) for a in R]))#best episode
        worst.append(min([sum(a) for a in R]))#worst episode
        #avr = sum([sum(a) for a in R])/len(R)
        avr = np.median([sum(a) for a in R])
        sum_each_epi = np.array([sum(a) for a in R])#total reward for each episode
        #avr_reward.append(int(avr))#
        avr_reward.append(math.floor(avr*10)/10)
        #m.append(np.mean(sum_each_epi))#mean of them
        m.append(np.median(sum_each_epi))
        #avr_var.append(sum([sum(a)/len(a) for a in variance])/len(variance))
        sq = 0#sum of (x-x_avr)**2
        for i in sum_each_epi:
            sq += (i - np.mean(sum_each_epi))**2
        sigma =((sq/len(sum_each_epi))**0.5)/(tra)**0.5
        upper.append(np.median(sum_each_epi)+2*sigma)
        lower.append(np.median(sum_each_epi)-2*sigma)
        np.save("m.npy",np.array(m))
        np.save("upper.npy",np.array(upper))
        np.save("lower.npy",np.array(lower))
        np.save("best.npy",np.array(best))
        np.save("worst.npy",np.array(worst))
        np.save("avr_reward.npy",np.array(avr_reward))
        return avr_reward,best,worst,avr,m,upper,lower
    def load_w(self):
        weights = torch.load("./weights.pt")
        size = (np.load("size.npy")).tolist()
        v = torch.load("./vless.pt")
        index = np.load("activation.npy",allow_pickle=True).item()
        var = float(np.load("./var.npy"))
        return weights,size,v,index, var
    def save_weights(self,weights,v,size,index,var_best):
        torch.save(weights,"./weights.pt")
        torch.save(v,"./vless.pt")
        np.save("size.npy",np.array(size))
        np.save("activation",index)
        np.save("var", var_best)
        return
