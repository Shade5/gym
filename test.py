import gym
from time import sleep
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW
import cv2
import numpy as np

env = gym.make('FetchPush-v1')
env.set_cameras(camera_names=['camera0', 'camera1', 'camera2'])
env.reset()
done = False
while True:
	obs, reward, done, info = env.step(env.action_space.sample())
	for k, v in info.items():
		if 'rgb' in k:
			cv2.imshow(k, cv2.cvtColor(np.flipud(v), cv2.COLOR_RGB2BGR))
		if 'depth' in k:
			cv2.imshow(k, (np.flipud(v) - v.min()) / (v.max() - v.min()))
	cv2.waitKey(1)
	# env.render()
	sleep(0.01)
