import numpy as np





def reward(state, action):
    ...

class Env:
    def __init__(self):
        self.start_state = np.array([0, 1, 0, -1, 0 , 0])
        self.mu = 0.055
        self.m = 1
        self.g = 9.81
        self.dt = 0.02
        
        self.state = self.start_state
        self.T_space = np.linspace(0, 2, 1000)
        self.phi_space = np.linspace(0, 2*np.pi, 1000)
        self.theta_space = np.linspace(0, np.pi, 1000)

    def get_next_state(self, action):
        T, phi, theta = action
        x, vx, y, vy, z, vz = self.state
        
        ax = (-1/np.sqrt(2)*np.cos(phi)*np.sin(theta) - 1/np.sqrt(2)*np.sin(phi))*T/self.m
        ay = (-1/np.sqrt(2)*np.cos(phi)*np.sin(theta) - 1/np.sqrt(2)*np.sin(phi))*T/self.m
        az = self.g - (np.cos(phi))*np.cos(theta)*T/self.m
    
        vx += (ax - self.mu * vx) * self.dt
        vy += (ay - self.mu * vy) * self.dt
        vz += (az - self.mu * vz) * self.dt
        
        x += vx * self.dt
        y += vy * self.dt
        z += vz * self.dt
        
        return np.array([x, vx, y, vy, z, vz])
    
    def reset(self):
        self.state = self.start_state
        return self.state