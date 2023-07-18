import numpy as np
import matplotlib.pyplot as plt

ld = [57, 62, 67, 72, 77]


fig = plt.figure()
ax1 = fig.add_subplot(121, projection ='3d')


l_est = [np.loadtxt(f'result/d{d}_est.csv', delimiter=" ") for d in ld]
l_gt = [np.loadtxt(f'result/d{d}gt.csv', delimiter=" ") for d in ld]

est = np.concatenate(l_est)
gt = np.concatenate(l_gt)

print(est.shape)

ax1.scatter(est[:,1], est[:,0],  est[:,2],  marker='^', c='red')
ax1.scatter(gt[:,1], gt[:,0], gt[:,2], marker='o', c='blue')
ax1.set_xlabel('x (cm)')
ax1.set_ylabel('y (cm)')
ax1.set_zlabel('z (cm)')
ax1.set_zlim([40, 80])

ax2 = fig.add_subplot(122)
ax2.scatter(est[:,1], est[:,0],  marker='^', c='red')
ax2.scatter(gt[:,1], gt[:,0],  marker='o', c='blue')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()