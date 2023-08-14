import gymnasium
from gym_examples.envs.grid_world import GridWorldEnv
env =  GridWorldEnv(render_mode = "human")
observation, info = env.reset(seed=42)
# wrapped_env = gymnasium.wrappers.FlattenObservation(env)
# print(wrapped_env.reset())
# wrap_env = gymnasium.wrappers.RelativePosition(env)
# print(wrap_env.reset())     # E.g.  [-3  3], {}

#previousInfo = -1
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   #if previousInfo == -1: previousInfo = info

   if terminated or truncated:
      observation, info = env.reset()


env.close()


#
# import gymnasium as gym
# from gymnasium.wrappers import GrayScaleObservation
# from gymnasium.wrappers import ResizeObservation
# #import cv2
# env = gym.make("LunarLander")
# # env = GrayScaleObservation(gym.make("LunarLander", render_mode="human"))
# env = ResizeObservation(env,64)
# observation, info = env.reset(seed=42)
#
# for _ in range(1000):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)
#
#    if terminated or truncated:
#       observation, info = env.reset()
#
# env.close()