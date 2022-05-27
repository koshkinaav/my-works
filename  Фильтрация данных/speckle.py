import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate
import json
from photutils.detection import DAOStarFinder

m = fits.open('speckledata.fits')[2].data
data = []
for i in m:
    data.append(np.array(Image.fromarray(i).resize((512, 512))))



def mean(data):
    mean = np.mean(data, axis = 0)
    plt.imshow(mean)
    plt.savefig('mean.png')
    return mean

def fourier(data):
    fourier = np.zeros((512, 512))
    for im in data:
         fourier += np.abs(np.fft.fftshift(np.fft.fft2(im)))**2
    fourier = fourier/len(fourier)
    plt.imshow(-fourier,cmap = 'Greys',vmin = int(-8*10**9))
    plt.savefig('fourier.png')
    return fourier

def masked():
    return np.array([np.array([True if (i - 256)**2 + (j-256)**2 <= 50**2 else False for j in range(512)]) for i in range(512)])

def mask(fourier):
    mask = masked()
    fourier -= fourier[~mask].mean()
    return fourier

def rotaver(fourier):
    angles  = np.linspace(0,100,100)
    rotaver = np.zeros(fourier.shape)
    for i in angles:
        rotaver += rotate(fourier,i,reshape = False)
    rotaver = rotaver/len(angles)
    plt.imshow(-rotaver,cmap = 'Greys', vmin = -10**9)
    plt.savefig('rotaver.png')
    return rotaver

def binary(fourier, rotaver):
    mask = masked()
    rotaver *= mask
    fourier *= mask
    binary = np.abs(np.fft.fftshift(np.fft.ifft2(np.divide(fourier,rotaver,out = np.zeros_like(fourier),where = rotaver!= 0))))
    plt.imshow(-binary, cmap = 'Greys')
    plt.tight_layout()
    plt.savefig('binary.png')
    return binary

def dist(binary):
    daofind = DAOStarFinder(fwhm = 3.0,threshold = 0.001)
    sources = daofind(binary)
    positions = np.transpose((sources['xcentroid'],sources['ycentroid']))
    s1,cent,s2 = positions
    dist = np.linalg.norm(((s2-cent)+(cent-s1))/2)*0.0206*(200/512.0)
    with open('binary.json','w') as f:
        json.dump({'distance': np.round(dist,3)},f)
    return dist
mean = mean(data)
fourier = mask(fourier(data))
rotaver = rotaver(fourier)
binary = binary(fourier,rotaver)
dist = dist(binary)
