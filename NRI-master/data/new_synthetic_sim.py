import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation

# Types of objects:
# red: attract each other
# blue: repell each other
# green: try to maintain a particular distance from each other (a spring b/t each one)
# yellow: centered on x or y axis



class DifferentParticlesSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
                 interaction_strength=1., noise_var=0.):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        arrs = np.eye(3)
        cols = np.random.choice(3,n_balls)
        self.colors = arrs[cols]

        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _get_unnattractive_force(self, loc_next):
        edges = np.ones((self.n_balls,self.n_balls))
        l2_dist_power3 = np.power(self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)

        forces_size = self.interaction_strength * edges / l2_dist_power3
        return forces_size

    def _get_attractive_force(self, loc_next):
        edges = -np.ones((self.n_balls,self.n_balls))
        l2_dist_power3 = np.power(self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)

        forces_size = self.interaction_strength * edges / l2_dist_power3
        return forces_size

    def _get_spring_force(self, loc_next):
        dists = self._l2(loc_next.transpose(), loc_next.transpose())

        forces_size = self.interaction_strength * (5 - dists)
        
        return forces_size

    def _get_forces(self, loc_next):
        n = self.n_balls
        outer = np.outer(self.colors.transpose(),self.colors.transpose())
        attract = outer[:n,:n]
        unnattract = outer[n:2*n,n:2*n]
        spring = outer[2*n:3*n,2*n:3*n]
        
        forces_size = (self._get_attractive_force(loc_next) * attract +
                        self._get_unnattractive_force(loc_next) * unnattract +
                        self._get_spring_force(loc_next) * spring
        
                        )
        np.fill_diagonal(forces_size, 0)  # self forces are zero (fixes division by zero)
        return forces_size

    def sample_trajectory(self, T=10000, sample_freq=10):
        n = self.n_balls
        
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            # half step leapfrog
            forces_size = self._get_forces(loc_next)

            # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = self._get_forces(loc_next)
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, self.colors


if __name__ == '__main__':
    sim = DifferentParticlesSim()

    t = time.time()
    loc, vel, colors = sim.sample_trajectory(T=5000, sample_freq=100)

    print(colors)
    print("Simulation time: {}".format(time.time() - t))
    vel_norm = np.sqrt((vel ** 2).sum(axis=1))
    # plt.figure()
    # axes = plt.gca()
    # axes.set_xlim([-5., 5.])
    # axes.set_ylim([-5., 5.])
    # for i in range(loc.shape[-1]):
    #     plt.plot(loc[:, 0, i], loc[:, 1, i])
    #     plt.plot(loc[0, 0, i], loc[0, 1, i], 'd')
    # plt.show()

    fig2 = plt.figure()
    axes = plt.gca()
    axes.set_xlim([-5., 5.])
    axes.set_ylim([-5., 5.])
    ims = []
        
    for i in range(loc.shape[0]):
        im = plt.plot(loc[i, 0, :], loc[i, 1, :], 'ro')
        ims.append(im)
    im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000)
    plt.show()