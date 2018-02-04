import numpy as np
#import math
"""
Contains the definition of the agent that will run in an
environment.
"""

class GreedyAgent:
    def __init__(self):
        self.Ntest=100
        self.count=1
        self.lst=[list() for i in range (10)]
        self.bestarm=0

    def act(self, observation):
        if (self.count>self.Ntest):
            return self.bestarm
        else:
            return int((self.count-1)/10)

    def reward(self, observation, action, reward):
        if self.count>self.Ntest:
            print(self.bestarm)
            pass
        elif self.count<self.Ntest:
            self.lst[action].append(reward)
        else:
            self.lst[action].append(reward)
            lstmp=[sum(item)/len(item) for item in self.lst]
            #print(lstmp)
            self.bestarm=lstmp.index(max(lstmp))

        self.count+=1

class OpEpGreedyAgent:
    def __init__(self):
        self.epsilon = 0.1
        self.Qlst = [[5] for i in range(10)]
        self.bestarm = 0

    def act(self, observation):
        tmp = np.random.random()
        if (tmp > self.epsilon):
            return self.bestarm
        else:
            return np.random.randint(0, 9)

    def reward(self, observation, action, reward):
        self.Qlst[action].append(reward)
        lstmp = [np.mean(item) for item in self.Qlst]
        self.bestarm = lstmp.index(max(lstmp))

class SoftmaxAgent:
    def __init__(self):
        self.freqlst = [1 for i in range(10)]
        self.rewardlst = [4.0 for i in range(10)]
        self.CDFlst = [ (i+1)/10.0 for i in range(10)]
        self.temperature=0.10

    def getProb(self,lst):
        tmp=[math.exp(i/self.temperature) for i in lst]
        ptmp=[i/sum(tmp) for i in tmp]
        return ptmp

    def act(self, observation):
        randomtmp = np.random.random()
        CDFarr=np.array(self.CDFlst)
        return int(np.where(CDFarr>=randomtmp)[0][0])

    def reward(self, observation, action, reward):
        self.rewardlst[action]=(self.rewardlst[action]*self.freqlst[action]+reward)/(int(self.freqlst[action])+1.0)
        self.freqlst[action]=int(1+self.freqlst[action])
        Ptmp=self.getProb(self.rewardlst)
        self.CDFlst =[sum(Ptmp[0:i]) for i in range(1,11)]

class UCBAgent:
    def __init__(self):
        self.time=1
        #self.freqlst=[ 1 for i in range(10)]
        self.freqlst = [0.0001 for i in range(10)]
        self.avgrewardlst = [2.0 for i in range(10)]
        self.target = [5.0 for i in range(10)]

    def act(self, observation):
        # this is the self.time_th time
        return int(self.target.index(max(self.target)))

    def reward(self, observation, action, reward):
        self.avgrewardlst[action]=(self.avgrewardlst[action]*self.freqlst[action]+reward)/(int(self.freqlst[action])+1.0)
        self.freqlst[action]=int(1+self.freqlst[action])
        self.time += 1
        self.target= [self.avgrewardlst[i]+math.sqrt(2*math.log(self.time)/self.freqlst[i]) for i in range(10)]

class UCBAgent_improved:
    def __init__(self):
        self.time=1.0
        self.freqlst = np.array([0.0001 for i in range(10)])
        self.avgrewardlst = np.array([2.0 for i in range(10)])
        self.target = np.array([5.0 for i in range(10)])

    def act(self, observation):
        # this is the self.time_th time
        return int(np.argmax(self.target))

    def reward(self, observation, action, reward):
        self.avgrewardlst[action]=(self.avgrewardlst[action]*self.freqlst[action]+reward)/(int(self.freqlst[action])+1.0)
        self.freqlst[action]=int(1+self.freqlst[action])
        self.time += 1
        self.target= self.avgrewardlst+np.sqrt(2*np.log(self.time)/self.freqlst)

Agent = UCBAgent_improved
