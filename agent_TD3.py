import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""

class ApproximateQ_Agent:
    def __init__(self):
        """Init a new agent.
        """
        self.lr = 0.3 # learning_rate, alpha
        self.gamma = 0.95   # reward_decay
        self.epsilon = 0.0    # e_greedy
        self.lamb_da=0.4

        self.n_x=150# i:0-4
        self.n_vx=50#j:0-4
        self.St = (-120, -1)  # x,vx
        self.At = -1
        self.Rtplus1 = 1

        self.weight = np.random.rand(self.n_x*self.n_vx, 3)
        self.eg = np.random.rand(self.n_x*self.n_vx, 3)
        #self.eg = np.zeros((self.n_x * self.n_vx, 3))

        minus_dmin=-150
        self.s=np.array([[minus_dmin-(i*minus_dmin/(self.n_x-1.0)) , -20.0+(j*40.0/(self.n_vx-1.0))] for i in range(self.n_x) for j in range(self.n_vx)])

        self.phi_St = self.getPhi(self.St)
        self.q_St = self.getQ_St(self.St,self.phi_St)
        self.T=0


    def reset(self, x_range):
        minus_dmin = x_range[0]
        self.T+=1
        #change s according to range
        self.s=np.array([[minus_dmin-(i*minus_dmin/(self.n_x-1.0)) , -20.0+(j*40.0/(self.n_vx-1.0))] for i in range(self.n_x) for j in range(self.n_vx)])


    def getQ_St(self, S_t,Phi):
        return np.dot(Phi, self.weight)

    def getPhi(self,S_t):
        return np.exp(-((S_t[0] - self.s[:, 0]) ** 2)) * np.exp(-(S_t[1] - self.s[:, 1]) ** 2)

    def act(self, observation):
        Phi_St=self.getPhi(observation)
        Q_St_p=self.getQ_St(observation,Phi_St)
        #update weights and eligibility trace
        sigma_t = self.Rtplus1 + self.gamma * np.max(Q_St_p) - self.q_St[self.At + 1]
        eg_tp=np.zeros((self.n_x*self.n_vx, 3))
        eg_tp[:,self.At+1]=self.phi_St
        eg_tp[:, np.argmax(Q_St_p)] -= self.gamma * Phi_St
        if np.argmax(Q_St_p)==self.At+1:
            self.eg=self.gamma*self.lamb_da*self.eg+eg_tp
        else:
            self.eg=eg_tp
        self.weight += self.lr * sigma_t * self.eg

        #use e-greedy for the first 180 games, and exploit for the last 20 games
        if self.T<=180 and np.random.uniform() <=self.epsilon:
            action = np.random.choice([-1, 0, 1])
        else:
            action = np.argmax(Q_St_p) - 1

        self.q_St=Q_St_p
        self.phi_St=Phi_St

        return action

    def reward(self, observation, action, reward):
        self.St=observation
        self.At=action
        self.Rtplus1=reward

Agent = ApproximateQ_Agent
