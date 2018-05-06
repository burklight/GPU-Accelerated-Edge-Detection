import numpy as np
import matplotlib.pyplot as plt

img = np.loadtxt('edge_detect.txt', delimiter = ',')
save = False if input("Save figure? ").lower() == 'no' else True
verbose = False if input("Plot figure? ").lower() == 'no' else True


fig, ax = plt.subplots(figsize=(15,15))
ax.imshow(img, cmap='gray')
if save == True:
    plt.savefig('result.png', format = 'png')
if verbose == True:
    plt.show()
