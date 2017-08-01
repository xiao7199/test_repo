import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# import Image
from PIL import Image
import os

#2D Gaussian function
def twoD_Gaussian((x, y), xo, yo, sigma_x, sigma_y):
    a = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    c = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    g = np.exp( - (a*((x-xo)**2) + c*((y-yo)**2)))
    return g.ravel()


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap


#Use base cmap to create transparent
mycmap = transparent_cmap(plt.cm.Reds)


# Import image and get x and y extents
I = Image.open('./GOPR032300000.jpg')
I = I.resize((640,480))
p = np.asarray(I).astype('float')
w, h = I.size
y, x = np.mgrid[0:h, 0:w]
labelfilename = './GOPR032300000.txt'
bbox_cnt = 0
key_point = []
if os.path.exists(labelfilename):
    with open(labelfilename) as f:
        for (i, line) in enumerate(f):
            if i == 0:
                bbox_cnt = int(line.strip())
                continue
            tmp = [int(t.strip()) for t in line.split()]
            key_point.append(tmp)

#Plot image and overlay colormap
fig, ax = plt.subplots(1, 1)
ax.imshow(I)
for pt in key_point:
    Gauss = twoD_Gaussian((x, y), pt[0], pt[1], .1*x.max(), .1*y.max())
    cb = ax.contourf(x, y, Gauss.reshape(x.shape[0], y.shape[1]), 15, cmap=mycmap)
plt.colorbar(cb)
plt.show()