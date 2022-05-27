import sys
import numpy as np
import json
from komm import BarkerSequence

filename = sys.argv[-1]
data = np.genfromtxt(filename)
def code_signal():
    code = BarkerSequence(11)
    seq = np.array(code.polar_sequence, dtype=np.int8)
    length = 5
    return np.repeat(seq,length)

code = code_signal()
data = np.convolve(data, code[::-1], mode='full')

std = np.std(data)
dta = []
for i in data:
    if i > 2 * std:
        dta.append(1)
    elif np.abs(i) < 2 * std:
        dta.append(0)
    else:
        dta.append(-1)


txt = []
for i in range(len(dta)):
    if dta[i] != dta[i - 1]:
       txt.append(dta[i])

txt = np.array([x for x in txt if x != 0])
txt = np.array((txt + 1) / 2, dtype=np.uint8)
txt = np.packbits(txt)
txt = txt.tobytes()
txt = txt.decode('ascii')

d = {"message": txt}

with open('wifi.json', 'w') as f:
    json.dump(d, f, indent=2)
