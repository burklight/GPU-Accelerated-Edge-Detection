import numpy as np
import matplotlib.pyplot as plt

save = False if input("Save figures? ").lower() == 'no' else True

timesG = np.loadtxt("./tests/speed_conv_img_size_GPU.txt", delimiter=",")
timesG = timesG.T
timesC = np.loadtxt("./tests/speed_conv_img_size_CPU.txt", delimiter=",")
timesC = timesC.T

speedup3_naive = (timesC[0] / timesG[0])
speedup3_naive = speedup3_naive[speedup3_naive > 0]
speedup3_shared = (timesC[0] / timesG[1])
speedup3_shared = speedup3_shared[speedup3_shared > 0]
speedup3_constant = (timesC[0] / timesG[2])
speedup3_constant = speedup3_constant[speedup3_constant > 0]
speedup3_separable = (timesC[0] / timesG[3])
speedup3_separable = speedup3_separable[speedup3_separable > 0]
speedup3_tiling = (timesC[0] / timesG[4])
speedup3_tiling = speedup3_tiling[speedup3_tiling > 0]

vals = ['256','512','1024','2048','4096']
dist = 1.0/7.0
xax = np.arange(5)
fig, ax = plt.subplots(figsize = (10,5))
plt.setp(ax, xticks=xax, xticklabels=vals)
ax.bar(xax-2*dist, speedup3_naive, width=dist, align = 'center', label = 'naive', edgecolor='black', color = 'white')
ax.bar(xax-dist, speedup3_shared, width=dist, align = 'center', label = 'shared',edgecolor='black', color = 'lightgray')
ax.bar(xax, speedup3_constant, width=dist, align = 'center', label = 'constant', edgecolor='black', color = 'darkgray')
ax.bar(xax+dist, speedup3_separable, width=dist, align = 'center', label = 'separable', edgecolor='black', color = 'gray')
ax.bar(xax+2*dist, speedup3_tiling, width=dist, align = 'center', label = 'adaptive tiling', edgecolor='black', color = 'black')
ax.legend()
ax.set_xlabel('Nx, Ny')
ax.set_ylabel('Speed-Up')
ax.set_title('5x5 convolution', fontsize=15)
if save == True:
    plt.savefig('./figures/experimental_su_5.png', format='png')
plt.show()

speedup5_naive = (timesC[1] / timesG[5])
speedup5_naive = speedup5_naive[speedup5_naive > 0]
speedup5_shared = (timesC[1] / timesG[6])
speedup5_shared = speedup5_shared[speedup5_shared > 0]
speedup5_constant = (timesC[1] / timesG[7])
speedup5_constant = speedup5_constant[speedup5_constant > 0]
speedup5_tiling = (timesC[1] / timesG[8])
speedup5_tiling = speedup5_tiling[speedup5_tiling > 0]

vals = ['256','512','1024','2048','4096']
dist = 1.0/7.0
xax = np.arange(5)
fig, ax = plt.subplots(figsize = (10,5))
plt.setp(ax, xticks=xax, xticklabels=vals)
ax.bar(xax-1.5*dist, speedup5_naive, width=dist, align = 'center', label = 'naive', edgecolor='black', color = 'white')
ax.bar(xax-0.5*dist, speedup5_shared, width=dist, align = 'center', label = 'shared',edgecolor='black', color = 'lightgray')
ax.bar(xax+0.5*dist, speedup5_constant, width=dist, align = 'center', label = 'constant', edgecolor='black', color = 'darkgray')
ax.bar(xax+1.5*dist, speedup5_tiling, width=dist, align = 'center', label = 'adaptive tiling', edgecolor='black', color = 'black')
ax.legend()
ax.set_xlabel('Nx, Ny')
ax.set_ylabel('Speed-Up')
ax.set_title('3x3 convolution', fontsize=15)
if save == True:
    plt.savefig('./figures/experimental_su_3.png', format='png')
plt.show()

vals = ['256','512','1024','2048','4096']
dist = 1.0/7.0
xax = np.arange(5)
fig, ax = plt.subplots(figsize = (10,5))
plt.setp(ax, xticks=xax, xticklabels=vals)
ax.bar(xax-2*dist, timesG[0], width=dist, align = 'center', label = 'naive', edgecolor='black', color = 'white')
ax.bar(xax-dist, timesG[1], width=dist, align = 'center', label = 'shared',edgecolor='black', color = 'lightgray')
ax.bar(xax, timesG[2], width=dist, align = 'center', label = 'constant', edgecolor='black', color = 'darkgray')
ax.bar(xax+dist, timesG[3], width=dist, align = 'center', label = 'separable', edgecolor='black', color = 'gray')
ax.bar(xax+2*dist, timesG[4], width=dist, align = 'center', label = 'adaptive tiling', edgecolor='black', color = 'black')
ax.legend()
ax.set_xlabel('Nx, Ny')
ax.set_ylabel('Time for 100 image filtering')
ax.set_title('5x5 convolution', fontsize=15)
if save == True:
    plt.savefig('./figures/experimental_rt_5.png', format='png')
plt.show()

vals = ['256','512','1024','2048','4096']
dist = 1.0/7.0
xax = np.arange(5)
fig, ax = plt.subplots(figsize = (10,5))
plt.setp(ax, xticks=xax, xticklabels=vals)
ax.bar(xax-1.5*dist, timesG[5], width=dist, align = 'center', label = 'naive', edgecolor='black', color = 'white')
ax.bar(xax-0.5*dist, timesG[6], width=dist, align = 'center', label = 'shared',edgecolor='black', color = 'lightgray')
ax.bar(xax+0.5*dist, timesG[7], width=dist, align = 'center', label = 'constant', edgecolor='black', color = 'darkgray')
ax.bar(xax+1.5*dist, timesG[8], width=dist, align = 'center', label = 'adaptive tiling', edgecolor='black', color = 'black')
ax.legend()
ax.set_xlabel('Nx, Ny')
ax.set_ylabel('Time for 100 image filtering')
ax.set_title('3x3 convolution', fontsize=15)
if save == True:
    plt.savefig('./figures/experimental_rt_3.png', format='png')
plt.show()
