import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from progressbar import ProgressBar
import mpl_toolkits.mplot3d.axes3d as p3
import scipy.signal as sig

class Unboxed_Diffusion():
    def __init__(self, Hyperparameters: dict):
        self.W = Hyperparameters['W']
        self.L = Hyperparameters['L']
        self.N = Hyperparameters['N']
        self.T = Hyperparameters['T']
        self.D = Hyperparameters['D']
        self.delt = Hyperparameters['delt']
        self.delx = Hyperparameters['delx']
        self.dely = Hyperparameters['dely']
        self.strip = Hyperparameters['strip']

        self.position_arr = np.zeros([self.N, int(self.T/self.delt), 2])

    
    def simulate(self):

        x1 = np.linspace(-self.strip, +self.strip, 10) # originallyy 10
        y1 = np.linspace(-self.W, +self.W, 10**4) # originally 10**4

        for i in range(self.N):
            self.position_arr[i,0,0] = x1[i%10] 
            self.position_arr[i,0,1] = y1[i%10**4]

        pbar = ProgressBar()
        for t in pbar(range(1,int(self.T/self.delt))):
            self.position_arr[:,t,0] = self.position_arr[:,t-1,0] + ((2*self.D*self.delt)**0.5)*np.random.normal(0,1,self.N)
            self.position_arr[:,t,1] = self.position_arr[:,t-1,1] + ((2*self.D*self.delt)**0.5)*np.random.normal(0,1,self.N)
        
        print("Simulation Complete")

    
    def plot_diff(self, time):
        fig, ax = plt.subplots()
        ax.scatter(self.position_arr[:,time,0], self.position_arr[:,time,1])
        ax.set_xlim(-self.L, +self.L)
        ax.set_ylim(-self.W, +self.W)
        ax.set_xlabel("x direction")
        ax.set_ylabel("y direction")
        plt.show()

    def plot_prob_3d(self, time, convolve = False, zlim = 15, save = False, name = "some_name",):
        xblocks = int(2*self.L/self.delx)
        yblocks = int(2*self.W/self.dely)

        xmesh = np.linspace(-self.L, +self.L, xblocks)
        ymesh = np.linspace(-self.W, +self.W, yblocks)
        xmesh, ymesh = np.meshgrid(xmesh, ymesh)

        prob = np.zeros([yblocks, xblocks])
        for i in range(self.N):
            x = int((self.position_arr[i,time,0]+self.L)/self.delx)
            y = int((self.position_arr[i,time,1]+self.W)/self.dely)
            if (x<xblocks and x>=0) and (y<yblocks and y>=0):
                prob[y,x] += 1 
        
        if convolve:
            # prob = convolve2D(prob, np.ones((3,3)) )
            f = 10
            prob = sig.convolve2d(prob, np.ones((f,f)), mode='same', boundary='fill', fillvalue=0)/f**2
            #prob = prob[1:-1, 1:-1]
        # print(prob.shape)

        fig = plt.figure()
        ax = p3.Axes3D(fig)
        # ax.contour3D(xmesh, ymesh, prob, 20, cmap='binary')
        ax.set_xlim(-self.L, +self.L)
        ax.set_ylim(-self.W, +self.W)
        ax.set_zlim(0, zlim)
        ax.plot_surface(xmesh, ymesh, prob,cmap='plasma', edgecolor='none')
        ax.set_xlabel("x direction")
        ax.set_ylabel("y direction")
        ax.set_zlabel("No of Points")
        if save:
            plt.savefig(name)
        return ax
    
    def get_prob_3d(self, xpoint, ypoint, time, convolve = True):
        xblocks = int(2*self.L/self.delx)
        yblocks = int(2*self.W/self.dely)

        xmesh = np.linspace(-self.L, +self.L, xblocks)
        ymesh = np.linspace(-self.W, +self.W, yblocks)
        xmesh, ymesh = np.meshgrid(xmesh, ymesh)

        prob = np.zeros([yblocks, xblocks])
        for i in range(self.N):
            x = int((self.position_arr[i,time,0]+self.L)/self.delx)
            y = int((self.position_arr[i,time,1]+self.W)/self.dely)
            if (x<xblocks and x>=0) and (y<yblocks and y>=0):
                prob[y,x] += 1 
        
        if convolve:
            # prob = convolve2D(prob, np.ones((3,3)) )
            f = 10
            prob = sig.convolve2d(prob, np.ones((f,f)), mode='same', boundary='fill', fillvalue=0)/f**2 
            
        x = int((xpoint+self.L)/self.delx)
        y = int((ypoint+self.W)/self.dely)
        
        return (prob[y,x], prob[y,x]/self.N)

    def animated_path(self, n, name = "plots/pathmoving.gif",):
        fig = plt.figure()
        ax = fig.gca()

        n_iter = 40

        print("starting pathmoving.gif")
        def a(iteration):
            ax.clear()
            xpoint = self.position_arr[n, iteration, 0]
            ypoint = self.position_arr[n, iteration, 1]
            
            for i in range(iteration):
                ax.plot(self.position_arr[n, i:i+2, 0], self.position_arr[n, i:i+2, 1], c = 'y')
            plt.xlim(-self.L, +self.L)
            plt.ylim(-self.W, +self.W)
            ax.scatter(xpoint,ypoint)
            ax.set_xlabel("x direction")
            ax.set_ylabel("y direction")
            
        animation_gd = animation.FuncAnimation(fig, a, frames=n_iter)
        animation_gd.save(name, writer="pillow", fps=5)
        print("done")
        plt.close()

    def animated_prob(self, time = 20):
        # Creates and stores multiple images of the probability distribution at different times
        for i in range(time):
            self.plot_prob_3d(i, convolve = True, save = True, name = "plots/prob_anim_unbox/probtime"+str(i)+".jpg")

        pass

    def animate_manypoints(self, nstart = 4975, nend = 5025, name = "plots/multi_point_paths_unobox.gif"):
        # Creates and stores multiple images of the probability distribution at different times
        fig = plt.figure()
        ax = fig.gca()

        n_iter = 40

        print("starting pathmoving.gif")
        def a(iteration):
            ax.clear()
            xpoint = self.position_arr[nstart:nend, iteration, 0]
            ypoint = self.position_arr[nstart: nend, iteration, 1]
            ax.set_xlim(-self.L, +self.L)
            ax.set_ylim(-self.W, +self.W)
            ax.scatter(xpoint,ypoint)
            ax.set_xlabel("x direction")
            ax.set_ylabel("y direction")
            
        animation_gd = animation.FuncAnimation(fig, a, frames=n_iter)
        animation_gd.save(name, writer="pillow", fps=5)
        print("done")
        plt.close()


class Boxed_Diffusion():
    def __init__(self, Hyperparameters: dict):
        self.W = Hyperparameters['W']
        self.L = Hyperparameters['L']
        self.N = Hyperparameters['N']
        self.T = Hyperparameters['T']
        self.D = Hyperparameters['D']
        self.delt = Hyperparameters['delt']
        self.delx = Hyperparameters['delx']
        self.dely = Hyperparameters['dely']
        self.strip = Hyperparameters['strip']

        self.position_arr = np.zeros([self.N, int(self.T/self.delt), 2])

    
    def simulate(self):

        x1 = np.linspace(-self.strip, +self.strip, 10) # originallyy 10
        y1 = np.linspace(-self.W, +self.W, 10**4) # originally 10**4

        for i in range(self.N):
            self.position_arr[i,0,0] = x1[i%10] 
            self.position_arr[i,0,1] = y1[i%10**4]

        pbar = ProgressBar()
        for t in pbar(range(1,int(self.T/self.delt))):
            xmov = 2*((2*self.D*self.delt)**0.5)*np.random.normal(0,1,self.N)
            ymov = 0.5*((2*self.D*self.delt)**0.5)*np.random.normal(0,1,self.N)
            self.position_arr[:,t,0] = np.where((self.position_arr[:,t-1,0] + xmov < self.L) & (self.position_arr[:,t-1,0] + xmov > -self.L), 
                                            self.position_arr[:,t-1,0] + xmov,
                                            self.position_arr[:,t-1,0] + 0.1*xmov)
            self.position_arr[:,t,1] = np.where((self.position_arr[:,t-1,1] + ymov < self.W) & (self.position_arr[:,t-1,1] + ymov > -self.W),
                                            self.position_arr[:,t-1,1] + ymov,
                                            self.position_arr[:,t-1,1] + 0.1*ymov)
        
        print("Simulation Complete")

        # print("Calculating Probabilities"), can add this part later

        # print("Done")
    
    def plot_diff(self, time):
        fig, ax = plt.subplots()
        ax.scatter(self.position_arr[:,time,0], self.position_arr[:,time,1])
        ax.set_xlim(-self.L, +self.L)
        ax.set_ylim(-self.W, +self.W)
        ax.set_xlabel("x direction")
        ax.set_ylabel("y direction")
        plt.show()

    def plot_prob_3d(self, time, convolve = False, save = False, name = "some_name", zlim = 15):
        xblocks = int(2*self.L/self.delx)
        yblocks = int(2*self.W/self.dely)

        xmesh = np.linspace(-self.L, +self.L, xblocks)
        ymesh = np.linspace(-self.W, +self.W, yblocks)
        xmesh, ymesh = np.meshgrid(xmesh, ymesh)

        prob = np.zeros([yblocks, xblocks])
        for i in range(self.N):
            x = int((self.position_arr[i,time,0]+self.L)/self.delx)
            y = int((self.position_arr[i,time,1]+self.W)/self.dely)
            if (x<xblocks and x>=0) and (y<yblocks and y>=0):
                prob[y,x] += 1 
        
        if convolve:
            # prob = convolve2D(prob, np.ones((3,3)) )
            f = 10
            prob = sig.convolve2d(prob, np.ones((f,f)), mode='same', boundary='fill', fillvalue=0)/f**2
            #prob = prob[1:-1, 1:-1]
        # print(prob.shape)

        fig = plt.figure()
        ax = p3.Axes3D(fig)
        # ax.contour3D(xmesh, ymesh, prob, 20, cmap='binary')
        ax.set_xlim(-self.L, +self.L)
        ax.set_ylim(-self.W, +self.W)
        ax.set_zlim(0, zlim)
        ax.plot_surface(xmesh, ymesh, prob,cmap='plasma', edgecolor='none')
        ax.set_xlabel("x direction")
        ax.set_ylabel("y direction")
        ax.set_zlabel("No of Points")
        if save:
            plt.savefig(name)
        return ax
    
    def get_prob_3d(self, xpoint, ypoint, time, convolve = True):
        xblocks = int(2*self.L/self.delx)
        yblocks = int(2*self.W/self.dely)

        xmesh = np.linspace(-self.L, +self.L, xblocks)
        ymesh = np.linspace(-self.W, +self.W, yblocks)
        xmesh, ymesh = np.meshgrid(xmesh, ymesh)

        prob = np.zeros([yblocks, xblocks])
        for i in range(self.N):
            x = int((self.position_arr[i,time,0]+self.L)/self.delx)
            y = int((self.position_arr[i,time,1]+self.W)/self.dely)
            if (x<xblocks and x>=0) and (y<yblocks and y>=0):
                prob[y,x] += 1 
        
        if convolve:
            # prob = convolve2D(prob, np.ones((3,3)) )
            f = 10
            prob = sig.convolve2d(prob, np.ones((f,f)), mode='same', boundary='fill', fillvalue=0)/f**2 
            
        x = int((xpoint+self.L)/self.delx)
        y = int((ypoint+self.W)/self.dely)
        
        return (prob[y,x], prob[y,x]/self.N)

    def animated_path(self, n, name = "plots/pathmoving.gif",):
        fig = plt.figure()
        ax = fig.gca()

        n_iter = 40

        print("starting pathmoving.gif")
        def a(iteration):
            ax.clear()
            xpoint = self.position_arr[n, iteration, 0]
            ypoint = self.position_arr[n, iteration, 1]
            
            for i in range(iteration):
                ax.plot(self.position_arr[n, i:i+2, 0], self.position_arr[n, i:i+2, 1], c = 'y')
            plt.xlim(-self.L, +self.L)
            plt.ylim(-self.W, +self.W)
            ax.set_xlabel("x direction")
            ax.set_ylabel("y direction")
            ax.scatter(xpoint,ypoint)
            
        animation_gd = animation.FuncAnimation(fig, a, frames=n_iter)
        animation_gd.save(name, writer="pillow", fps=5)
        print("done")
        plt.close()

    def animated_prob(self, time = 20):
        # Creates and stores multiple images of the probability distribution at different times
        for i in range(time):
            self.plot_prob_3d(i, convolve = True, save = True, name = "plots/prob_anim/probtime"+str(i)+".jpg")

        pass

    def animate_manypoints(self, nstart = 4975, nend = 5025, name = "plots/multi_point_paths.gif"):
        # Creates and stores multiple images of the probability distribution at different times
        fig = plt.figure()
        ax = fig.gca()

        n_iter = 40

        print("starting pathmoving.gif")
        def a(iteration):
            ax.clear()
            xpoint = self.position_arr[nstart:nend, iteration, 0]
            ypoint = self.position_arr[nstart: nend, iteration, 1]
            ax.set_xlim(-self.L, +self.L)
            ax.set_ylim(-self.W, +self.W)
            ax.scatter(xpoint,ypoint)
            ax.set_xlabel("x direction")
            ax.set_ylabel("y direction")
            
        animation_gd = animation.FuncAnimation(fig, a, frames=n_iter)
        animation_gd.save(name, writer="pillow", fps=5)
        print("done")
        plt.close()