from time import sleep
from mujoco_py import GlfwContext
import gym
GlfwContext(offscreen=True)  # Create a window to init GLFW
import cv2
import numpy as np
from transforms3d.euler import quat2mat
from mayavi.mlab import points3d, show
import pickle

width, height = 128, 128
fovy = 45

env = gym.make('FetchPush-v1')
env.set_cameras(width=width, height=height, camera_names=['camera0', 'camera1', 'camera2'])
env.reset()

f = 0.5 * height / np.tan(fovy * np.pi / 360)
K = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))

t0 = env.sim.data.get_camera_xpos("camera0")[:, None]
t1 = env.sim.data.get_camera_xpos("camera1")[:, None]
t2 = env.sim.data.get_camera_xpos("camera2")[:, None]

# Mujoco uses -Z axis a camera front, need to fix that
x_rot = quat2mat([0, 1, 0, 0])
R0 = env.sim.data.get_camera_xmat("camera0") @ x_rot
R1 = env.sim.data.get_camera_xmat("camera1") @ x_rot
R2 = env.sim.data.get_camera_xmat("camera2") @ x_rot

dict_to_save = {'intrinsics': K,
                'extrinsics': [np.vstack((np.hstack((R0, t0)), [0, 0, 0, 1])),
                               np.vstack((np.hstack((R1, t1)), [0, 0, 0, 1])),
                               np.vstack((np.hstack((R2, t2)), [0, 0, 0, 1]))],
                'rgb': [],
                'depth': []}
for i in range(3):
    env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        dict_to_save['rgb'].append(np.stack([info['rgb_' + c] for c in ['camera0', 'camera1', 'camera2']]))
        dict_to_save['depth'].append(np.stack([info['depth_' + c] for c in ['camera0', 'camera1', 'camera2']]))
        for k, v in info.items():
            if 'rgb' in k:
                cv2.imshow(k, cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
            if 'depth' in k:
                cv2.imshow(k, (v - v.min()) / (v.max() - v.min()))
        cv2.waitKey(1)
        sleep(0.01)

with open('fetch_data.pkl', 'wb') as handle:
    pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # View camera locations
# P = np.array([[0, 0, 0],
#               [-0.1, -0.1, 0.1],
#               [-0.1, 0.1, 0.1],
#               [0.1, -0.1, 0.1],
#               [0.1, 0.1, 0.1]]).T
#
# C0 = R0 @ P + t0
# points3d(C0[0], C0[1], C0[2], color=(1, 0, 0), scale_factor=0.05)
#
# C1 = R1 @ P + t1
# points3d(C1[0], C1[1], C1[2], color=(0, 1, 0), scale_factor=0.05)
#
# C2 = R2 @ P + t2
# points3d(C2[0], C2[1], C2[2], color=(0, 0, 1), scale_factor=0.05)
# show()