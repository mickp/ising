""" Simple 2D Ising model """

import numpy as np
from matplotlib import pyplot, animation
import zlib

kB = 1.3806488e-23 # Joules / Kelvin

bound = lambda iterable, N: tuple(0 if val < 0 else N if val > N 
                             else val for val in iterable)
shift = lambda p, q: map(lambda a, b: a + b, p, q)

class IsingRunner():
    def __init__(self, **kwargs):
        def figupdate(*args):
            model.iterate()
            img.set_array(model.lattice)
            fig.canvas.set_window_title('T=%sK; count=%s' % (model.T, args[0]))
            return img,
        model = Ising2d(**kwargs)
        self.model = model
        fig, ax = pyplot.subplots(1,1)
        img = ax.imshow(model.lattice)
        img.set_interpolation('none')
        ani = animation.FuncAnimation(fig, figupdate, interval=50, blit=True)
        ani.save_count = 0
        fig.show()


class Ising2d(object):
    def __init__(self, N=32, Jv=1e-20, Jh=1e-20, T=300, mu=0, dT=0, percyc=0):
        self.N = N
        self.Jv = Jv
        self.Jh = Jh
        self.T = T
        self.mu = mu
        self.dT = dT
        self.percyc = percyc

        self.lattice = np.random.choice([-1,1], (N,N), int)

        self.histT = []
        self.histS = []

    def iterate(self):
        self.T = max(self.T + self.dT, 1e-6)
        N = self.N
        lattice = self.lattice
        neighbours = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        coeffs = [.25 * self.Jv, .25 * self.Jh, .25 * self.Jv, .25 * self.Jh]
        mu = 0.5 * self.mu

        n = self.percyc or (N * N / 4)
        p = np.random.random_integers(0, N-1, n)
        q = np.random.random_integers(0, N-1, n)

        ## Evaluating np.exp(- (newH - H) / (kB * self.T)).
        # Use a hash, instead.
        p_flip = {-8: np.exp(8 / (kB * self.T)),
                  -4: np.exp(4 / (kB * self.T)),
                   0: 1.0,
                   4: np.exp(-4 / (kB * self.T)),
                   8: np.exp(-8 / (kB * self.T))}

        for i in range(n):
            index = (p[i], q[i])
            H = mu * lattice[index] \
                + sum([-J * lattice[index]
                          * lattice[bound(shift(index, delta), N - 1)]
                          for delta, J in zip(neighbours, coeffs)])
            newH = mu * -lattice[index] \
                   + sum([-J * -lattice[index]
                             * lattice[bound(shift(index, delta), N - 1)]
                             for delta, J in zip(neighbours, coeffs)])
            if newH <= H:
                lattice[index] *= -1
            #elif np.exp(- (newH - H) / (kB * self.T)) >= np.random.rand():
            elif p_flip[int(round(newH - H))] > np.random.rand():
                lattice[index] *= -1

        self.histT.append(self.T)
        self.histS.append(len(zlib.compress(lattice)))
