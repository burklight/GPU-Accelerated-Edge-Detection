import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

save = False if input("Save figures? ").lower() == 'no' else True

tac = 0.234
tag = 0.456
tl = 0.0125
tw = 0.0125
bw = 80

tmpMx = np.arange(2, 10)
tmpMy = np.arange(2, 10)
Msx, Msy = np.meshgrid(tmpMx, tmpMy)

# Shared Memory

def AI_shared(Mx, My):
    return (2*Mx * My) / (4.0 * 4.0 + 4.0 * Mx * My)

def perf_shared(Mx, My):
    return AI_shared(Mx, My) * bw

Pers = np.zeros((len(tmpMx),len(tmpMy)))
for i in range(8):
    for j in range(8):
        Pers[i,j] = perf_shared(tmpMx[i], tmpMy[j])

fig = plt.figure(figsize=(20,10))
ax = fig.gca(projection='3d')
ax.set_xlabel("Mx")
ax.set_ylabel("My")
ax.set_zlabel("Performance (GFlops/s)")
surf = ax.plot_surface(Msx, Msy, Pers, cmap="viridis")
fig.colorbar(surf)
if save == True:
    plt.savefig("../figures/theoretical_shared_performance.png",format="png")
plt.show()

# Constant Memory

def AI_constant(Mx, My):
    return (2*Mx * My) / (4.0 * 4.0)

def perf_constant(Mx, My):
    return AI_constant(Mx, My) * bw

Pers = np.zeros((len(tmpMx),len(tmpMy)))
for i in range(8):
    for j in range(8):
        Pers[i,j] = perf_constant(tmpMx[i], tmpMy[j])

fig = plt.figure(figsize=(20,10))
ax = fig.gca(projection='3d')
ax.set_xlabel("Mx")
ax.set_ylabel("My")
ax.set_zlabel("Performance (GFlops/s)")
surf = ax.plot_surface(Msx, Msy, Pers, cmap="viridis")
fig.colorbar(surf)
if save == True:
    plt.savefig("../figures/theoretical_constant_performance.png",format="png")
plt.show()

# Adaptive tiling (1x21 tiling)

def AI_adaptive(Mx, My, T):
    return (2*Mx * My * T) / (4.0*(2.0 * 4.0 + (T-2)*2.0))

def perf_adaptive(Mx, My,T):
    return AI_adaptive(Mx, My,T) * bw

tiling = 21
Pers = np.zeros((len(tmpMx),len(tmpMy)))
for i in range(8):
    for j in range(8):
        Pers[i,j] = perf_adaptive(tmpMx[i], tmpMy[j], tiling)

fig = plt.figure(figsize=(20,10))
ax = fig.gca(projection='3d')
ax.set_xlabel("Mx")
ax.set_ylabel("My")
ax.set_zlabel("Performance (GFlops/s)")
surf = ax.plot_surface(Msx, Msy, Pers, cmap="viridis")
fig.colorbar(surf)
if save == True:
    plt.savefig("../figures/theoretical_tiling_performance.png",format="png")
plt.show()
