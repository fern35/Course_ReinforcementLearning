import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""

class ActorCritic_Agent:
    def __init__(self):
        """Init a new agent.
        """
        self.lr = 0.5 # learning_rate
        self.gamma = 0.99   # reward_decay
        self.deviation_coeff = 30.0 # the coefficient related to the deviation of the distribution of the policy

        self.n_x=70# i:0-4
        self.n_vx=40#j:0-4

        self.weight1_mu = np.random.rand(self.n_x*self.n_vx)# for the policy

        self.weight2 = np.random.rand(self.n_x*self.n_vx)# for the value function

        minus_dmin=-150
        self.s=np.array([[minus_dmin-(i*minus_dmin/(self.n_x-1.0)) , -20.0+(j*40.0/(self.n_vx-1.0))]
                         for i in range(self.n_x) for j in range(self.n_vx)])
        self.Time = 0
        self.St = (-100, 0)  # x,vx
        self.At = 0
        self.Rtplus1 = 0
        self.mu= self.get_frombases(self.weight1_mu,self.getPhi(self.St))
        self.deviation = 10.
        self.Phi_St=self.getPhi(self.St)
        self.I=1
        self.game=0# the number of games


    def reset(self, x_range):
        self.game +=1
        self.Time = 1.0
        self.I = 1
        minus_dmin = x_range[0]
        #change s(discretization of x and vx) according to range
        self.s=np.array([[minus_dmin-(i*minus_dmin/(self.n_x-1.0)) , -20.0+(j*40.0/(self.n_vx-1.0))]
                         for i in range(self.n_x) for j in range(self.n_vx)])
        self.eg1_mu = np.zeros(self.n_x * self.n_vx)
        self.eg2 = np.zeros(self.n_x * self.n_vx)

    def get_frombases(self, weight, Phi):
        return np.dot(Phi, weight)

    def getPhi(self,S_t):
        return np.exp(-(S_t[0] - self.s[:, 0]) ** 2) * np.exp(-(S_t[1] - self.s[:, 1]) ** 2)

    def act(self, observation):

        self.Time += 1.0
        #update the policy/value function for the last time
        Phi_Stplus1=self.getPhi(observation)
        self.dt = self.Rtplus1 + self.gamma * self.get_frombases(self.weight2,Phi_Stplus1)\
                  -self.get_frombases(self.weight2,self.Phi_St)
        # update the policy
        logPi_gradient_mu= ((self.At-self.mu)/(self.deviation**2))*self.Phi_St
        self.weight1_mu += self.lr*self.dt*self.I*logPi_gradient_mu
        #update the value function
        self.weight2 += self.lr*self.dt*self.I*self.Phi_St

        self.I*=self.gamma


        # act
        #update Phi_St
        self.Phi_St=Phi_Stplus1
        # generate the mean and devistion for the policy
        self.mu= self.get_frombases(self.weight1_mu,self.Phi_St)
        self.deviation = self.deviation_coeff/np.log(self.Time)

        if self.game>180:
            action=self.mu
        else:
            action=np.random.normal(self.mu, self.deviation)
        return action

    def reward(self, observation, action, reward):
        self.St=observation
        self.At=action
        self.Rtplus1=reward


class ActorCritic_Agent1:
    def __init__(self):
        """Init a new agent.
        """
        self.lr = 0.02 # learning_rate
        self.gamma = 0.99   # reward_decay
        self.deviation_coeff = 15.0 # the coefficient related to the deviation of the distribution of the policy

        self.n_x=30# i:0-4
        self.n_vx=15#j:0-4
        self.temp_x=self.n_x# parameter for adjusting the features
        self.temp_vx=self.n_vx # parameter for adjusting the features

        self.weight1_mu = np.random.rand(self.n_x*self.n_vx)# for the policy

        self.weight2 = np.random.rand(self.n_x*self.n_vx)# for the value function

        minus_dmin=-150
        self.s=np.array([[minus_dmin-(i*minus_dmin/(self.n_x-1.0)) , -20.0+(j*40.0/(self.n_vx-1.0))]
                         for i in range(self.n_x) for j in range(self.n_vx)])
        self.Time = 0
        self.St = (-100, 0)  # x,vx
        self.At = 0
        self.Rtplus1 = 0
        self.mu= self.get_frombases(self.weight1_mu,self.getPhi(self.St))
        self.deviation = 10.
        self.Phi_St=self.getPhi(self.St)
        self.I=1
        self.game=0# the number of games


    def reset(self, x_range):
        self.game +=1
        self.Time = 1.0
        self.I = 1
        minus_dmin = x_range[0]
        #change s(discretization of x and vx) according to range
        self.s=np.array([[minus_dmin-(i*minus_dmin/(self.n_x-1.0)) , -20.0+(j*40.0/(self.n_vx-1.0))]
                         for i in range(self.n_x) for j in range(self.n_vx)])
        self.eg1_mu = np.zeros(self.n_x * self.n_vx)
        self.eg2 = np.zeros(self.n_x * self.n_vx)

    def get_frombases(self, weight, Phi):
        return np.dot(Phi, weight)

    def getPhi(self,S_t):
        return np.exp(-(S_t[0] - self.s[:, 0]) ** 2/self.temp_x) * np.exp(-(S_t[1] - self.s[:, 1]) ** 2/self.temp_vx)

    def act(self, observation):

        self.Time += 1.0
        #update the policy/value function for the last time
        Phi_Stplus1=self.getPhi(observation)
        self.dt = self.Rtplus1 + self.gamma * self.get_frombases(self.weight2,Phi_Stplus1)\
                  -self.get_frombases(self.weight2,self.Phi_St)
        # update the policy
        logPi_gradient_mu= ((self.At-self.mu)/(self.deviation**2))*self.Phi_St
        self.weight1_mu += self.lr*self.dt*self.I*logPi_gradient_mu
        #update the value function
        self.weight2 += self.lr*self.dt*self.I*self.Phi_St

        self.I*=self.gamma


        # act
        #update Phi_St
        self.Phi_St=Phi_Stplus1
        # generate the mean and devistion for the policy
        self.mu= self.get_frombases(self.weight1_mu,self.Phi_St)
        self.deviation = self.deviation_coeff/np.log(self.Time)

        if self.game>180:
            action=self.mu
        else:
            action=np.random.normal(self.mu, self.deviation)
        return action

    def reward(self, observation, action, reward):
        self.St=observation
        self.At=action
        self.Rtplus1=reward

Agent = ActorCritic_Agent1
