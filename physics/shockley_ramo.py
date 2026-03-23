import numpy as np
import numba as nb
from numba import njit
from scipy import integrate, interpolate, signal

from .interpolation import linear_interp_field
from .transportation import mobility


"""
This module computes induced signals using the Shockley–Ramo theorem
and convolves them with an electronics transfer function.
"""


class SignalProcessing:
    def __init__(self, te, ta, td, elec):
        """
        Parameters:
        -----------
        te : float
            Time step for interpolation / simulation
        ta : float
            Total acquisition time
        td : array
            Drift time array
        elec : str
            Electronics type ("top" or "bot")
        """
        self.te = te
        self.ta = ta
        self.td = td
        self.elec = elec

    def select_electronics(self):
        """
        Load the electronics response and return:
        - relative time axis
        - amplitude response
        - sampling time
        """
        if self.elec == "top":
            electronics = np.loadtxt("/Users/pinchault/Documents/work/research/DUNE_steak/steak/electronics/top.txt")
            reltime, ampl = electronics[:, 0], electronics[:, 1]
            sampling = sampling_top

        elif self.elec == "bot":
            electronics = np.loadtxt("/Users/pinchault/Documents/work/research/DUNE_steak/steak/electronics/bottom.txt")
            reltime, ampl = electronics[:, 0], electronics[:, 1]
            sampling = sampling_bot

        else:
            raise ValueError("Unknown electronics type")

        return reltime, ampl, sampling

    def convolved(self, S, norm=True):
        """
        Convolve the signal S with the electronics response.

        Parameters:
        -----------
        S : array
            Input signal
        norm : bool
            Normalize transfer function if True

        Returns:
        --------
        tc : array
            Time axis after convolution
        Sc : array
            Convolved signal
        """
        reltime, ampl, _ = self.select_electronics()

        # Resample transfer function to match simulation time step
        reltime_interp = np.arange(0, np.max(reltime), self.te, dtype=np.float32)
        response_interp = interpolation(reltime_interp, reltime, ampl)

        # Normalize transfer function
        if norm:
            response_interp /= integrate.trapezoid(response_interp, reltime_interp)

        print(f"Signal size: {len(S)} -- Response size: {len(response_interp)}")

        # Time axis after convolution
        tc = np.linspace(
            0,
            self.ta + reltime_interp[-1],
            len(self.td) + len(reltime_interp) - 1
        )

        # Perform convolution (normalized by time step)
        Sc = self.te * signal.fftconvolve(S, response_interp, "full")

        return tc, Sc

    def daq(self, tc, Sc):
        """
        Simulate data acquisition system (DAQ):
        - resampling
        - charge integration
        """
        _, _, sampling = self.select_electronics()

        # Electronics sampling time axis
        sample_rate = np.arange(0, max(tc) + sampling, sampling)

        print(f"Number of samples: {len(sample_rate)}")

        # Interpolate signal on sampling grid
        S_sampling = interpolation(sample_rate, tc, Sc)

        # Integrate signal to get charge
        Q = integrate.cumulative_trapezoid(S_sampling, sample_rate)

        # Compute discrete charge differences
        deltaQ = Q[1:] - Q[:-1]

        print(f"deltaQ size: {len(deltaQ)}")

        return sample_rate[:-2], deltaQ


class SimulationParams:
    def __init__(self, ne, td, x, y, z):
        """
        Parameters:
        -----------
        ne : int
            Number of electrons
        td : array
            Time array
        x, y, z : arrays
            Electron trajectories
        """
        self.ne = ne
        self.td = td
        self.x = x
        self.y = y
        self.z = z

    def charge(self, s):
        """
        Compute total charge per electron by integrating signal.
        """
        q = np.zeros(self.ne, dtype=np.float32)

        for e in range(self.ne):
            q[e] = integrate.trapezoid(s[e], self.td)

        return q

    def drift_time_anode(self, xs):
        """
        Compute drift time between shielding plane and anode.
        """
        t_drift = np.zeros(self.ne, dtype=np.float32)

        for e in range(self.ne):
            x_drift = self.x[e][self.x[e] != 0]

            t_coll = self.td[len(x_drift)]

            idx_shield = np.where(self.x[e] >= xs)
            t_shield = self.td[idx_shield[0][0]]

            t_drift[e] = t_coll - t_shield

        return t_drift

    def start_point(self):
        """
        Get starting positions of electrons.
        """
        xi = np.array([self.x[e][0] for e in range(self.ne)], dtype=np.float32)
        yi = np.array([self.y[e][0] for e in range(self.ne)], dtype=np.float32)
        zi = np.array([self.z[e][0] for e in range(self.ne)], dtype=np.float32)

        return xi, yi, zi

    def end_point(self):
        """
        Get final positions of electrons.
        """
        xf = np.zeros(self.ne, dtype=np.float32)
        yf = np.zeros(self.ne, dtype=np.float32)
        zf = np.zeros(self.ne, dtype=np.float32)

        for e in range(self.ne):
            mask = self.x[e] != 0
            xf[e] = self.x[e][mask][-1]
            yf[e] = self.y[e][mask][-1]
            zf[e] = self.z[e][mask][-1]

        return xf, yf, zf


@nb.jit(nopython=True, parallel=True)
def induced_signal(ne, n_step, vol, x, y, z, vx, vy, vz,
                   ew_view0, ew_view1, ew_view2, step_x):
    """
    Compute induced signals for all electrons in parallel.
    """
    Sind1 = np.zeros((ne, n_step), dtype=np.float32)
    Sind2 = np.zeros((ne, n_step), dtype=np.float32)
    S_view2 = np.zeros((ne, n_step), dtype=np.float32)

    for e in nb.prange(ne):
        # WARNING: print inside numba slows everything
        Sind1[e], Sind2[e], S_view2[e] = ramo(
            vol, x[e], y[e], z[e],
            vx[e], vy[e], vz[e],
            ew_view0, ew_view1, ew_view2,
            step_x, n_step
        )

    return Sind1, Sind2, S_view2


@njit
def ramo(vol, x, y, z, vx, vy, vz,
         Ew0, Ew1, Ew2, step_x, n_step):
    """
    Compute instantaneous current using Shockley–Ramo theorem.
    """
    q = -1.0  # electron charge (normalized)

    ind1 = np.zeros(n_step, dtype=np.float32)
    ind2 = np.zeros(n_step, dtype=np.float32)
    coll = np.zeros(n_step, dtype=np.float32)

    for idx in range(n_step):

        i = int(np.round(x[idx] / step_x))
        j = int(np.round(y[idx] / step_x))
        k = int(np.round(z[idx] / step_x))

        # Interpolate weighting fields
        Exw0, Eyw0, Ezw0 = linear_interp_field(
            x[idx], y[idx], z[idx],
            vol, i, j, k,
            step_x, step_x,
            Ew0[0], Ew0[1], Ew0[2]
        )

        Exw1, Eyw1, Ezw1 = linear_interp_field(
            x[idx], y[idx], z[idx],
            vol, i, j, k,
            step_x, step_x,
            Ew1[0], Ew1[1], Ew1[2]
        )

        Exw2, Eyw2, Ezw2 = linear_interp_field(
            x[idx], y[idx], z[idx],
            vol, i, j, k,
            step_x, step_x,
            Ew2[0], Ew2[1], Ew2[2]
        )

        # Shockley–Ramo current
        ind1[idx] = q * (vx[idx] * Exw0 + vy[idx] * Eyw0 + vz[idx] * Ezw0)
        ind2[idx] = q * (vx[idx] * Exw1 + vy[idx] * Eyw1 + vz[idx] * Ezw1)
        coll[idx] = q * (vx[idx] * Exw2 + vy[idx] * Eyw2 + vz[idx] * Ezw2)

    return ind1, ind2, coll


def interpolation(te, t, f):
    """
    1D linear interpolation with extrapolation.
    """
    interp_func = interpolate.interp1d(t, f, fill_value="extrapolate")
    return interp_func(te)


# Electronics parameters
ADC = 256
sampling_top = 0.5     # µs
sampling_bot = 0.512   # µs