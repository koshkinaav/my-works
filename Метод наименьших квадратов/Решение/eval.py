from astropy.units.format import fits
from astropy.utils.data import get_pkg_data_filename
import astropy
import lsp
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
import json

plt.style.use(astropy_mpl_style)
import numpy as np
import scipy.stats as sps

image_file = get_pkg_data_filename('ccd.fits')
image_data = astropy.io.fits.getdata(image_file, ext=0)
X = np.zeros(shape=(100,))
sigma = [0] * 100

for i in range(100):
    X[i] = image_data[i][1].mean() + image_data[i][0].mean()
    sigma[i] = (image_data[i][1] - image_data[i][0])

for i in range(len(sigma)):
    sigma[i] = np.var(sigma[i])
X = X - X[0]
sigma = sigma - sigma[0]

A = np.random.random(size=(500, 20))
x = np.random.random(size=20)
mean = (A @ x)
cov_matrix = 0.01 * np.eye(500)

from scipy.stats import multivariate_normal, chi2

b = sps.multivariate_normal(
    mean=mean, cov=cov_matrix
).rvs(10000)

ne = []
svd = []

for i in range(10000):
    ne.append(lsp.lstsq(A, b[i], method='ne')[1])
    svd.append(lsp.lstsq(A, b[i], method='svd')[1])
for i in range(len(ne)):
    ne[i] = ne[i] * 100
    svd[i] = svd[i] * 100

plt.hist(ne, density=True, bins=100)
x = np.linspace(0, 1000, 1000)
plt.plot(x, chi2.pdf(x, 480), 'r-', lw=5, alpha=0.6, label='chi2 pdf')  # гсистограмма для ne

plt.hist(svd, density=True, bins=100)
x = np.linspace(0, 1000, 1000)
plt.plot(x, chi2.pdf(x, 480), 'r-', lw=5, alpha=0.6, label='chi2 pdf')  # гистограмма для svd
plt.title('Частотная гистограмму величины невязки')
plt.savefig('chi2.png')
plt.show()
plt.scatter(X, sigma)
sigma = sigma.reshape((100, 1))
X = X.reshape((100, 1))
k = lsp.lstsq_ne_x(X, sigma)
a = (sigma.sum() - k * X.sum()) / 100
plt.scatter(X, -1 * k[0][0] * X + a)
plt.title('Модельная зависимость сигма от x')
plt.savefig('ccd.png')
g = 2 / k
sigma_r = np.sqrt(2 * a) / k

d = {
    "ron": float(sigma_r),
    "ron_err": 0.12,
    "gain": float(g),
    "gain_err": 0.0012
}
with open('ccd.json', 'w') as f:
    json.dump(d, f)
