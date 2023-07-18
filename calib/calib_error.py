#Get intrinsic matrix: rs-enumerate-devices -c >> file.txt
#  Intrinsic of "Color" / 640x480 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y16}
#   Width:      	640
#   Height:     	480
#   PPX:        	322.131988525391
#   PPY:        	243.162139892578
#   Fx:         	614.8515625
#   Fy:         	615.16162109375
#   Distortion: 	Inverse Brown Conrady
#   Coeffs:     	0  	0  	0  	0  	0  
#   FOV (deg):  	54.99 x 42.62

import cv2
import numpy as np
import matplotlib.pyplot as plt

distortion = np.array([0, 0, 0, 0, 0])
intrinsic = np.array([[614.8515625, 0.0,              322.1319885253906], 
                      [0.0,         615.16162109375,  243.16213989257812],
                      [0.0,         0.0,              1.0]])

#''' 
ds_use_at_72cm_extrinsic = np.array([[.62, 0.33], [.67, 0.38]])  #s = d*x + y
ab = [[ds_use_at_72cm_extrinsic[-1][0], 1],[ds_use_at_72cm_extrinsic[-2][0], 1]]
c = [ds_use_at_72cm_extrinsic[-1][1], ds_use_at_72cm_extrinsic[-2][1]]
ret = np.linalg.solve(ab, c)
d = np.linspace(0.5, 0.8, 10)
print('ret = ', ret)
# exit()
'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(ds_at_diff_depth_extrinsic[:,0], ds_at_diff_depth_extrinsic[:,1], marker='o')
# ax1.plot(ds_use_at_72cm_extrinsic[:,0], ds_use_at_72cm_extrinsic[:,1], marker='^')
ax1.plot(d, s, marker='*')
ax1.set_xlabel('depth (m)')
ax1.set_ylabel('scale')
ax1.legend(['diff extrinisc', 'same extrinsic', 'Ground-truth'])
'''
# plt.show()
#'''
#exit()
base = 0.67
b_load_intrinsic = True
b_load_extrinsic = True
distance = 67
# scale = 0.33
scale = (distance/100)*ret[0] + ret[1]
print('scale = ', scale)

uvd_xyz_files = [f"calib_d0.{distance}.csv"]
l_arr = [np.loadtxt(file, delimiter=",", skiprows=1, dtype=np.float32) for file in uvd_xyz_files]

def cal_delta(arr):
    mu_depth = np.mean(arr[:,2])
    mu_x = np.mean(arr[:,3])
    delta = mu_x - mu_depth
    # print(f'mu_depth1 = {mu_depth}, mu_x1={mu_x}, delta1={delta}')
    return [delta, mu_depth, mu_x]
mu_depth = [cal_delta(arr)[1] for arr in l_arr]
mu_x = [cal_delta(arr)[2] for arr in l_arr]
delta = np.array(mu_x) - np.array(mu_depth)
m_delta = np.mean(delta)
# print(mu_x)
# print(mu_depth)
# print(delta)
print(f'm_delta = ', m_delta)
#exit()

def filter_out(arr, threshold=1.117):
    l = []
    for e in arr:
        if e[5] > threshold:
            l.append(e)
    arr = np.array(l)
    #print('l = ', arr.shape)
    return arr

l_len_arr = []
for i in range(len(l_arr)):
    # print('\nl_arr[i] = ', l_arr[i].shape)
    l_arr[i] = filter_out(l_arr[i])
    # print('l_arr[i] = ', l_arr[i].shape)
    l_len_arr.append(l_arr[i].shape[0])

print('l_len_arr = ', l_len_arr)
#exit()
arr = np.concatenate(l_arr)
print('arr = ', arr.shape)
np_img =  arr[:,0:2].astype('float32')
np_obj = arr[:,3:6].astype('float32')

# swap x and z 
np_obj[:, [2, 0]] = np_obj[:, [0, 2]]

mtx, dist = intrinsic, distortion 
for i in range(1):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([np_obj], [np_img], (640, 480), mtx, None, None, None, cv2.CALIB_USE_INTRINSIC_GUESS)
intrinsic, distortion = mtx, dist

if not b_load_intrinsic:
    print('Save intrinsic parameters...')
    np.save('param/intrinic.npy', intrinsic)
    np.save('param/dis.npy', distortion)
else:
    print('Load parameters...')
    intrinsic = np.load('param/intrinic.npy')
    distortion = np.load('param/dis.npy')

_, rvec1, tvec1 = cv2.solvePnP(np_obj, np_img, intrinsic, distortion, flags=cv2.SOLVEPNP_UPNP, useExtrinsicGuess=True)
R_mtx, jac = cv2.Rodrigues(rvec1)
tran = tvec1


if not b_load_extrinsic:
    print('Save extrinsic parameters...')
    np.save('param/R_mtx.npy', R_mtx)
    np.save('param/tran.npy', tran)
else:
    print('Load  extrinsic parameters...')
    R_mtx = np.load('param/R_mtx.npy')
    tran = np.load('param/tran.npy')

tran[1][0] = tran[1][0] + (base - distance/100)/4

print('Intrinsic: \n', intrinsic)
print('Distortion: \n', distortion)
print('R_mtx: \n', R_mtx)
print('tran: \n', tran)


inv_intrinsic = np.linalg.pinv(intrinsic)
inv_rotation = np.linalg.pinv(R_mtx)


errors = []
errors_x = []
errors_y = []
errors_z = []
# f = open("3d_calib.txt", "w")

est = []
gt = []
for entry in range(np_obj.shape[0]):
    u, v = np_img[entry][0], np_img[entry][1]
    uv_1 = np.array([[u,v,1]], dtype=np.float32)
    uv_1 = uv_1.T
    suv_1 = scale*uv_1
    xyz_c = inv_intrinsic.dot(suv_1)
    xyz_c = xyz_c - tran
    XYZ = inv_rotation.dot(xyz_c)

    #Re-adjust depth
   
    if 0 <= entry and entry < l_len_arr[0]:
        XYZ[2][0] = mu_x[0]
    '''
    elif  l_len_arr[0] <= entry and entry < (l_len_arr[0]+ l_len_arr[1]):
        XYZ[2][0] = mu_x[1]
    elif  l_len_arr[1] <= entry and entry < (l_len_arr[1]+ l_len_arr[2]):
        XYZ[2][0] = mu_x[2]
    '''

    # print(f"Pixel Coor: u = {u}, v = {v}")
    # print(f"Estimation: x = {XYZ[0][0]:.4f}, y = {XYZ[1][0]:.4f}, z = {XYZ[2][0]:.4f}")
    est.append([XYZ[0][0], XYZ[1][0], XYZ[2][0]])
    gt.append([np_obj[entry][0], np_obj[entry][1], np_obj[entry][2]])
    # print(f"Ground Truth: x = {np_obj[entry][0]:.4f}, y = {np_obj[entry][1]:.4f}, z = {np_obj[entry][2]:.4f}")
    # f.write(f"{XYZ[0][0]:.10f} {XYZ[1][0]:.10f} {XYZ[2][0]:.10}\n")

    errors_x.append(abs(XYZ[0][0] - np_obj[entry][0]))
    errors_y.append(abs(XYZ[1][0] - np_obj[entry][1]))
    errors_z.append(abs(XYZ[2][0] - np_obj[entry][2]))

    err_dist = np.linalg.norm(np.array(XYZ.flatten()) - np_obj[entry])
    errors.append(err_dist)
    # print(f"Error Distance = {err_dist:.4f}")

    # print("======================================================")

# f.close()

'''
print(f"ERROR SUMMARY:")
print(f">> Minimum Distance Error = {min(errors):.4f}")
print(f">> Maximum Distance Error = {max(errors):.4f}")
print(f">> Average Distance Error = {np.mean(errors):.4f}")
print("")
print(f">> Minimum Error x-axis = {min(errors_x):.4f}")
print(f">> Maximum Error x-axis = {max(errors_x):.4f}")
print(f">> Average Error x-axis = {np.mean(errors_x):.4f}")
print("")
print(f">> Minimum Error y-axis = {min(errors_y):.4f}")
print(f">> Maximum Error y-axis = {max(errors_y):.4f}")
print(f">> Average Error y-axis = {np.mean(errors_y):.4f}")
print("")
print(f">> Minimum Error z-axis = {min(errors_z):.4f}")
print(f">> Maximum Error z-axis = {max(errors_z):.4f}")
print(f">> Average Error z-axis = {np.mean(errors_z):.4f}")
'''

est = np.array(est)*100
gt = np.array(gt)*100
print(est.shape, gt.shape)

fig = plt.figure()
ax1 = fig.add_subplot(121, projection ='3d')

ax1.scatter(est[:,1], est[:,0],  est[:,2],  marker='^', c='red')
ax1.scatter(gt[:,1], gt[:,0], gt[:,2], marker='o', c='blue')
ax1.set_xlabel('x (cm)')
ax1.set_ylabel('y (cm)')
ax1.set_zlabel('z (cm)')
ax1.set_zlim([40, 80])

np.savetxt(f'result/d{distance}_est.csv', est)
np.savetxt(f'result/d{distance}gt.csv', gt)

ax2 = fig.add_subplot(122)

# l1 = l_len_arr[0]
# l2 = l_len_arr[1] + l1
# l3 = l_len_arr[2] + l2
# ax2.scatter(est[l1:l2,1], est[l1:l2,0],  marker='^', c='red')
# ax2.scatter(gt[l1:l2,1], gt[l1:l2,0],  marker='o', c='blue')

ax2.scatter(est[:,1], est[:,0],  marker='^', c='red')
ax2.scatter(gt[:,1], gt[:,0],  marker='o', c='blue')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()