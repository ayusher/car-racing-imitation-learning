import tensorflow as tf
import gym
import numpy as np
import random

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

model = tf.keras.models.load_model("model")
giffer = []
env = gym.make("CarRacing-v0")
for i_episode in range(1):
    env.seed(int(random.random()*100))
    observation = env.reset()
    old_s = rgb2gray(observation)
    last6 = [old_s for _ in range(6)]
    while True:
        env.render()
        last6.pop(0)
        giffer.append(observation)
        last6.append(rgb2gray(observation))
        action = model.predict(np.array([np.dstack(tuple(last6))])/255.0)[0]
        observation, reward, done, info = env.step(action)
        if done:
            print("Terminal reward {}".format(reward))
            break
env.close()

import imageio
import cv2

for i in range(len(giffer)):
    giffer[i] = cv2.resize(giffer[i], dsize = (400, 400), interpolation=cv2.INTER_NEAREST)
imageio.mimsave("vid.gif", giffer, fps=24)
