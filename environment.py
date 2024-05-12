import numpy as np

class Environment(object):
    def __init__(self, state, mu, m, g, thetamin = 0, thetamax = np.pi, phimin = 0, phimax = 2*np.pi, Tmin = 0, Tmax = 20, dT = 1, dt = 0.02, dphi = 1/np.pi, dtheta = 1/np.pi):
        self.mu = mu
        self.m = m
        self.g = g
        self.thetamin = thetamin
        self.thetamax = thetamax
        self.phimin = phimin
        self.phimax = phimax
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.dT = dT
        self.dt = dt
        self.dtheta = dtheta
        self.dphi = dphi
        self.state = state

    def reset(self):
        self.state = np.array([0, 1, 0, -1, 0, 0])
    
    def next_state(self, action):
        x, vx, y, vy, z, vz = self.state
        T, phi, theta = action

        ax = (-0.7071 * np.cos(phi) * np.sin(theta) - 0.7071 * np.sin(phi)) * T / self.m
        ay = ax
        az = self.g - (np.cos(phi) * np.cos(theta)) * T / self.m

        vxn = vx + (ax - self.mu * vx) * (self.dt / 2)
        vyn = vy + (ay - self.mu * vy) * (self.dt / 2)
        vzn = vz + (az - self.mu * vz) * (self.dt / 2)

        xn = x + vxn * self.dt
        yn = y + vyn * self.dt
        zn = z + vzn * self.dt

        vxn = vx + (ax - self.mu * vx) * (self.dt / 2)
        vyn = vy + (ay - self.mu * vy) * (self.dt / 2)
        vzn = vz + (az - self.mu * vz) * (self.dt / 2)

        self.state = np.array([xn, vxn, yn, vyn, zn, vzn])
        done = zn < -25
        
        return [[x, vx, y, vy, z, vz],[xn, vxn, yn, vyn, zn, vzn], done]


    def reward1(self):
        x, vx, y, vy, z, vz = self.state
        
        reward1 = -(np.sqrt((x - 5)**2 + y**2 + (z + 20)**2))
        if(reward1 == 0):
            reward1 += 100000
            done = True
        return reward1
    
    def reward2(self, current_step, alpha, beta):
        x, vx, y, vy, z, vz = self.state
        
        reward2 = - (alpha*(np.sqrt((x - 5*np.cos(1.2*(current_step)*self.dt))**2 + (y - 5 * np.sin(1.2*(current_step)*self.dt))**2 + (z + 20)**2)) + beta*(np.sqrt((vx + 6*np.sin(1.2*(current_step)*self.dt))**2 + (vy - 6*np.cos(1.2*(current_step)*self.dt))**2 + vz**2)) )
        
        return reward2
    
    def actionspace(self):
        num_steps_T = int((self.Tmax - self.Tmin) / self.dT) + 1
        num_steps_phi = int((self.phimax - self.phimin) / self.dphi) + 1
        num_steps_theta = int((self.thetamax - self.thetamin) / self.dtheta) + 1


        T_values = [self.Tmin + i * self.dT for i in range(num_steps_T)]
        phi_values = [self.phimin + i * self.dphi for i in range(num_steps_phi)]
        theta_values = [self.thetamin + i * self.dtheta for i in range(num_steps_theta)]


        action_space = []
        for dT in T_values:
            for dphi in phi_values:
                for dtheta in theta_values:
                    action_space.append([dT, dphi, dtheta])
        return np.array(action_space, dtype=np.float32)


        
    
