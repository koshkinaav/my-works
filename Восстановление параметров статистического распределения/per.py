import matplotlib.pyplot as plt
import numpy as np
import random
import math
import json
from scipy.optimize import minimize
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

center_coord = SkyCoord('02h21m00s +57d07m42s')
vizier = Vizier(
    columns=['RAJ2000', 'DEJ2000', 'pmRA', 'pmDE'],
    column_filters={'BPmag': '<16', 'pmRA': '!=', 'pmDE': '!='},  # число больше — звёзд больше
    row_limit=10000
)
stars = vizier.query_region(
    center_coord,
    width=1.0 * u.deg,
    height=1.0 * u.deg,
    catalog=['I/350'],  # Gaia EDR3
)[0]

ra = stars['RAJ2000']._data  # прямое восхождение, аналог долготы
dec = stars['DEJ2000']._data  # склонение, аналог широты
x1 = (ra - ra.mean()) * np.cos(dec / 180 * np.pi) + ra.mean()
x2 = dec
v1 = stars['pmRA']._data
v2 = stars['pmDE']._data
x = np.vstack([x1, x2, v1, v2])


def t(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2):
    x = np.array(x)
    tau0 = 1.0 - tau1 - tau2
    T0 = tau0 / (2 * np.pi * sigma02) * np.exp(-0.5 * np.sum(x[2:] ** 2, axis=0) / sigma02)
    l = 1.0 / (2 * np.pi * sigmav2) * np.exp(-0.5 * np.sum((x[2:] - muv.reshape(-1, 1)) ** 2, axis=0) / sigmav2)
    T1 = tau1 / (2 * np.pi * sigmax2) * np.exp(-0.5 * np.sum((x[:2] - mu1.reshape(-1, 1)) ** 2, axis=0) / sigmax2) * l
    T2 = tau2 / (2 * np.pi * sigmax2) * np.exp(-0.5 * np.sum((x[:2] - mu2.reshape(-1, 1)) ** 2, axis=0) / sigmax2) * l
    T = T0 + T1 + T2

    T0 = np.divide(T0, T, out=np.full_like(T, 1.0 / 3), where=T != 0)
    T1 = np.divide(T1, T, out=np.full_like(T, 1.0 / 3), where=T != 0)
    T2 = np.divide(T2, T, out=np.full_like(T, 1.0 / 3), where=T != 0)
    tau1 = np.sum(T1) / np.sum(T0 + T1 + T2)
    tau2 = np.sum(T2) / np.sum(T0 + T1 + T2)
    muv = np.sum((T1 + T2) * x[2:], axis=1) / np.sum(T1 + T2)
    mu1 = np.sum(T1 * x[:2], axis=1) / np.sum(T1)
    mu2 = np.sum(T2 * x[:2], axis=1) / np.sum(T2)
    sigma02 = np.sum(T0 * np.sum(x[2:] ** 2, axis=0)) / np.sum(T0) / 2
    sigmax2 = np.sum(T1 * np.sum((x[:2] - mu1.reshape(-1, 1)) ** 2, axis=0) +
                     T2 * np.sum((x[:2] - mu2.reshape(-1, 1)) ** 2, axis=0)) / np.sum(T1 + T2) / 2
    sigmav2 = np.sum((T1 + T2) * np.sum((x[2:] - muv.reshape(-1, 1)) ** 2, axis=0)) / np.sum(T1 + T2) / 2
    return (tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2)


def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2):
    th = (tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2)
    for i in range(100):
        th = t(x, *th)

    return (th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7])


tau1_init = 0.1
tau2_init = 0.2
muv_init = np.median(x[2:], axis=1)
sigmax2_init = np.var(x[:2])
sigmav2_init = np.var(x[2:])
mu1_init = np.median(x[:2], axis=1) - np.array([0.25, 0.25])
mu2_init = np.median(x[:2], axis=1) + np.array([0.25, 0.25])
result = em_double_cluster(x, tau1_init, tau2_init, muv_init, mu1_init, mu2_init, 10 * sigmav2_init, sigmax2_init,
                           sigmav2_init)
motion = result[2]
center1 = result[3]
center2 = result[4]
sigma0 = result[5]
sigmax2 = result[6]
sigmav2 = result[7]
size_ratio = result[0] / result[1]
j = {
    "size_ratio": size_ratio,
    "motion": {"ra": float(motion[0]), "dec": float(motion[1])},
    "clusters": [
        {
            "center": {"ra": float(center1[0]), "dec": float(center1[1])},
        },
        {
            "center": {"ra": float(center2[0]), "dec": float(center2[1])},
        }
    ]
}

with open('per.json', 'w') as f:
    json.dump(j, f)
