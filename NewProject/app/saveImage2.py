import numpy as np
import matplotlib.pyplot as plt

img = np.loadtxt('edge_detect.txt', delimiter = ',')
path = input()

fig, ax = plt.subplots(figsize=(15,15))
ax.imshow(img, cmap='gray')
plt.savefig(path, format = 'png')
