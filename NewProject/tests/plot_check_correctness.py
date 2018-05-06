import numpy as np
import matplotlib.pyplot as plt


image_cpu = np.loadtxt('./cpu_edge_detect.txt', delimiter=',', dtype = int)
fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(image_cpu, cmap="gray")
ax.set_title('CPU edge detection', fontsize=30)
plt.show()
del(image_cpu)

image_gpu_naive = np.loadtxt('./gpu_naive_edge_detect.txt', delimiter=',', dtype = int)
fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(image_gpu_naive, cmap="gray")
ax.set_title('GPU edge detection (naive approach)', fontsize =30)
plt.show()
del(image_gpu_naive)

image_gpu_shared = np.loadtxt('./gpu_shared_edge_detect.txt', delimiter=',', dtype = int)
fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(image_gpu_shared, cmap="gray")
ax.set_title('GPU edge detection (shared memory approach)', fontsize =30)
plt.show()
del(image_gpu_shared)

image_gpu_const = np.loadtxt('./gpu_const_edge_detect.txt', delimiter=',', dtype = int)
fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(image_gpu_const, cmap="gray")
ax.set_title('GPU edge detection (shared memory + constant kernels approach)', fontsize =30)
plt.show()
del(image_gpu_const)

image_gpu_sep = np.loadtxt('./gpu_sep_edge_detect.txt', delimiter=',', dtype = int)
fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(image_gpu_sep, cmap="gray")
ax.set_title('GPU edge detection (separable filters + shared memory + constant kernels approach)', fontsize =30)
plt.show()
del(image_gpu_sep)

image_gpu_tiling = np.loadtxt('./gpu_tiling_edge_detect.txt', delimiter=',', dtype = int)
fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(image_gpu_tiling, cmap="gray")
ax.set_title('GPU edge detection (separable filters + shared memory + constant kernels + tiling approach)', fontsize =30)
plt.show()
del(image_gpu_tiling)
