import numpy as np

class Environment(object):
    def __init__(self, state, mu, m, g, thetamin = 0, thetamax = np.pi, phimin = 0, phimax = 2*np.pi, Tmin = 0, Tmax = 20, dt = 0.02, dphi = 1/np.pi, dtheta = 1/np.pi):
        self.mu = mu
        self.m = m
        self.g = g
        self.thetamin = thetamin
        self.thetamax = thetamax
        self.phimin = phimin
        self.phimax = phimax
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.dt = dt
        self.dtheta = dtheta
        self.dphi = dphi
        self.state = state

    def reset(self):
        self.state = np.array([0, 1, 0, -1, 0, 0])
    
    def infostep(self, current_step, action):
        x, vx, y, vy, z, vz = self.state
        T, phi, theta = action
        
        ax = (-0.7071 * np.cos(phi) * np.sin(theta) - 0.7071 * np.sin(phi)) * T / self.m
        ay = (-0.7071 * np.cos(phi) * np.sin(theta) - 0.7071 * np.sin(phi)) * T / self.m
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

        # reward=0.5* -np.sqrt((x - 5 * np.cos(1.2 * self.current_step * self.dt))**2 + (y - 5 * np.sin(1.2 * self.current_step * self.dt))**2 + (z + 20)**2)+0.3 * np.sqrt((vx+6 * np.sin(1.2 * self.current_step * self.dt))**2 + (vy-6*np.cos(1.2 * self.current_step * self.dt))**2 + vz**2)
        reward = -np.sqrt((xn - 5 * np.cos(1.2 * current_step * self.dt))**2 + (yn - 5 * np.sin(1.2 * current_step * self.dt))**2 + (zn + 20)**2)
        done = zn < -25 or current_step > (self.Tmax / self.dt)
        self.state = np.array([xn, vxn, yn, vyn, zn, vzn])
        return [[xn, vxn, yn, vyn, zn, vzn], reward, done]
    
    def actionspace(self):
        num_steps_T = int((self.Tmax - self.Tmin) / self.dt) + 1
        num_steps_phi = int((self.phimax - self.phimin) / self.dphi) + 1
        num_steps_theta = int((self.thetamax - self.thetamin) / self.dtheta) + 1


        T_values = [self.Tmin + i * self.dt for i in range(num_steps_T)]
        phi_values = [self.phimin + i * self.dphi for i in range(num_steps_phi)]
        theta_values = [self.thetamin + i * self.dtheta for i in range(num_steps_theta)]


        action_space = []
        for dT in T_values:
            for dphi in phi_values:
                for dtheta in theta_values:
                    action_space.append([dT, dphi, dtheta])
        return np.array(action_space, dtype=np.float32)


        
    
