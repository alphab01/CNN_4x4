import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
def sigm(x):
    return 1 / (1 + np.exp(-x))
def softma(x):
    return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
def deriv(x):
    return x * (1 - x)
def gener(q):
    a = []
    for i in range(q):
        a.append(random.randint(0, 1))
    b = []
    b.append(a)
    return np.array(b)
inp = np.array([[0, 0, 0, 1,
                  0, 0, 1, 1,
                  0, 0, 0, 1,
                  0, 0, 1, 1],
                 [0, 0, 1, 0,
                  0, 1, 1, 0,
                  0, 0, 1, 0,
                  0, 1, 1, 1],
                 [0, 1, 0, 0,
                  1, 1, 0, 0,
                  0, 1, 0, 0,
                  1, 1, 1, 0],
                 [0, 0, 1, 0,
                  0, 1, 0, 1,
                  0, 1, 0, 1,
                  0, 0, 1, 0],
                 [0, 1, 0, 0,
                  1, 0, 1, 0,
                  1, 0, 1, 0,
                  0, 1, 0, 0],
                 [0, 1, 0, 0,
                  1, 1, 0, 0,
                  0, 1, 0, 0,
                  1, 1, 1, 0],
                 [0, 0, 1, 0,
                  0, 1, 1, 0,
                  0, 0, 1, 0,
                  0, 1, 1, 1],
                 [0, 1, 1, 1,
                  0, 1, 0, 1,
                  0, 1, 0, 1,
                  0, 1, 1, 1],
                 [1, 1, 1, 0,
                  1, 0, 1, 0,
                  1, 0, 1, 0,
                  1, 1, 1, 0]])
g = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])
for ok in range(50):
    np.append(inp, gener(16))
    np.append(g, [0, 0, 1])
tinp = np.array([[1, 1, 1, 1,
                  1, 0, 0, 1,
                  1, 0, 0, 1,
                  1, 1, 1, 1],
                 [0, 0, 1, 0,
                  0, 1, 1, 0,
                  0, 0, 1, 0,
                  0, 0, 1, 0]])
tg = np.array([[1, 0, 0], [0, 1, 0]])
def get_img(l, rf, rt, cf, ct):
    sub = l[:, rf:rt, cf:ct]
    return sub.reshape(-1, 1, rt-rf, ct-cf)
batch = 1
img = inp.reshape(-1, 4, 4)
timg = tinp.reshape(-1, 4, 4)
lab = g
np.random.seed(1)
a = 0.00005
kr = 2
kc = 2
nk = 8
kernels = 0.2 * np.random.random((kr * kc, nk)) - 0.1
inpr = 4
inpc = 4
orow = inpr - kr + 1
ocol = inpc - kc + 1
h = orow * ocol * nk
w12 = 0.2 * np.random.random((h, 3)) - 0.1
for i in range(1000000):
    for j in range(int(len(img)/batch)):
        bs = j * batch
        be = bs + batch
        l0 = img[bs:be]
        sects = list()
        for r in range(orow):
            for c in range(ocol):
                sect = get_img(l0, r, r + kr, c, c + kc)
                sects.append(sect)
        l0_resh = np.concatenate(sects, axis = 1)
        l0_flat = l0_resh.reshape(-1, kr * kc)
        ko = l0_flat.dot(kernels)
        l1 = sigm(ko.reshape(batch, -1))
        l2 = softma(l1.dot(w12))
        bat_lab = lab[bs:be]
        l2d = (bat_lab - l2)/batch
        l1d = l2d.dot(w12.T) * deriv(l1)
        w12 += a * l1.T.dot(l2d)
        l1d_resh = l1d.reshape(-1, nk)
        kernels += a * l0_flat.T.dot(l1d_resh)
    if (i%1000 == 0):
        err = 0
        for k in range(int(len(tg)/batch)):
            bs = k * batch
            be = bs + batch
            l0 = timg[bs:be]
    
            sects = list()
            for r in range(orow):
                for c in range(ocol):
                    sect = get_img(l0, r, r + kr, c, c + kc)
                    sects.append(sect)
            l0_resh = np.concatenate(sects, axis = 1)
            l0_flat = l0_resh.reshape(-1, kr * kc)
            ko = l0_flat.dot(kernels)
            l1 = sigm(ko.reshape(batch, -1))
            l2 = softma(l1.dot(w12))
            for ki in range(len(l2[0])):
                err += (l2[0][ki] - tg[k][ki])**2
        print(err, i)
    '''if (i%10 == 0):
        print(np.round(err, 2))'''
bs = 0 * batch
be = bs + batch
l0 = img[bs:be]
sects = list()
for r in range(orow):
    for c in range(ocol):
        sect = get_img(l0, r, r + kr, c, c + kc)
        sects.append(sect)
l0_resh = np.concatenate(sects, axis = 1)
l0_flat = l0_resh.reshape(-1, kr * kc)
ko = l0_flat.dot(kernels)
l1 = sigm(ko.reshape(batch, -1))
l2 = softma(l1.dot(w12))
bat_lab = lab[bs:be]
print(l2)
c = int(input())
d = 0
fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize=(20, 8))
ax = ax.flatten()
while (d < 10):
    q = gener(16)
    q = q.reshape(-1, 4, 4)
    l0 = q[0:1]
    sects = list()
    for r in range(orow):
        for cc in range(ocol):
            sect = get_img(l0, r, r + kr, cc, cc + kc)
            sects.append(sect)
    l0_resh = np.concatenate(sects, axis = 1)
    l0_flat = l0_resh.reshape(-1, kr * kc)
    ko = l0_flat.dot(kernels)
    l1 = sigm(ko.reshape(batch, -1))
    l2 = softma(l1.dot(w12))
    bat_lab = lab[bs:be]
    z = 0
    o = 0
    z += l2[0][0]
    o += l2[0][1]
    if (z > o and c == 0 and z > l2[0][2] and z > 0.99):
        im = ax[d].imshow(q[0], cmap = 'viridis', aspect = 'auto')
        d += 1
    elif (o > z and c == 1 and o > l2[0][2] and o > 0.99):
        im = ax[d].imshow(q[0], cmap = 'viridis', aspect = 'auto')
        d += 1
plt.show()
