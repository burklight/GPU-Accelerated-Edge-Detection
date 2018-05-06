import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(20,20))

image_cpu = np.loadtxt('./tests/cpu_edge_detect.txt', delimiter=',', dtype = int)
axes[0,0].imshow(image_cpu, cmap="gray")
axes[0,0].set_title('CPU edge detection')
#plt.show()
del(image_cpu)

image_gpu_naive = np.loadtxt('./tests/gpu_naive_edge_detect.txt', delimiter=',', dtype = int)
axes[0,1].imshow(image_gpu_naive, cmap="gray")
axes[0,1].set_title('GPU edge detection (naive approach)')
#plt.show()
del(image_gpu_naive)

image_gpu_shared = np.loadtxt('./tests/gpu_shared_edge_detect.txt', delimiter=',', dtype = int)
axes[0,2].imshow(image_gpu_shared, cmap="gray")
axes[0,2].set_title('GPU edge detection (shared memory)')
#plt.show()
del(image_gpu_shared)

image_gpu_const = np.loadtxt('./tests/gpu_const_edge_detect.txt', delimiter=',', dtype = int)
axes[1,0].imshow(image_gpu_const, cmap="gray")
axes[1,0].set_title('GPU edge detection (constant kernels)')
#plt.show()
del(image_gpu_const)

image_gpu_sep = np.loadtxt('./tests/gpu_sep_edge_detect.txt', delimiter=',', dtype = int)
axes[1,1].imshow(image_gpu_sep, cmap="gray")
axes[1,1].set_title('GPU edge detection (separable filters)')
#plt.show()
del(image_gpu_sep)

image_gpu_tiling = np.loadtxt('./tests/gpu_tiling_edge_detect.txt', delimiter=',', dtype = int)
axes[1,2].imshow(image_gpu_tiling, cmap="gray")
axes[1,2].set_title('GPU edge detection (adaptive tiling)')
plt.show()
del(image_gpu_tiling)
