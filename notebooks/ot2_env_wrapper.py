import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    X_MIN, X_MAX = 0.10775, 0.253
    Y_MIN, Y_MAX = 0.088, 0.2195
    Z_MIN, Z_MAX = 0.1695, 0.2897
    MAX_STEPS = 1000

    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define action and observation space
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Keep track of the number of steps
        self.steps = 0

        # Initialize goal position
        self.goal_position = np.zeros(3, dtype=np.float32)

        # Initialize pipette position
        self.pipette_position = np.zeros(3, dtype=np.float32)

    def reset(self, seed=42):
        # Set a seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # Set a random goal position within the working area
        self.goal_position = np.random.uniform(
            low=[self.X_MIN, self.Y_MIN, self.Z_MIN],
            high=[self.X_MAX, self.Y_MAX, self.Z_MAX],
        )

        # Call the environment reset function
        states = self.sim.reset()
        self.robotid = list(states.keys())[0]
        self.pipette_position = np.array(
            states.get(self.robotid, {}).get('pipette_position', []),
            dtype=np.float32,
        )
        self.observation = np.concatenate(
            [self.pipette_position, self.goal_position], dtype=np.float32
        )

        # Reset the number of steps
        self.steps = 0

        # Initialize pipette position
        print(self.observation)
        return self.observation, {}

    def step(self, action, drop=0):
        # Execute one time step within the environment
        # Append 0 for the drop action
        action = np.append(action, drop)

        # Call the environment step function
        self.observation = self.sim.run([action])

        # Process the observation
        self.pipette_position = np.array(
            self.observation[self.robotid]['pipette_position'],
            dtype=np.float32,
        )
        # Calculate the reward and termination status
        reward, terminated = self.calculate_reward(self.pipette_position)

        # Convert the observation to a flat array
        self.observation = np.concatenate(
            [self.pipette_position, self.goal_position], dtype=np.float32
        )

        # Check if the episode should be truncated
        truncated = self.steps == self.MAX_STEPS
        if truncated:
            reward += reward * 5

        # Print relevant information for debugging
        # print(f"Step: {self.steps}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, "
        #       f"Pipette Position: {self.pipette_position}, Goal Position: {self.goal_position}")

        info = {}  # No additional information to return

        # Increment the number of steps
        self.steps += 1

        return self.observation, reward, terminated, truncated, info

    def calculate_reward(self, pipette_position):
        # Calculate the negative distance reward
        negative_distance_reward = -np.linalg.norm(
            self.pipette_position - self.goal_position
        )

        # Add a small positive reward for getting closer to the goal
        proximity_reward = (
            10  # You can adjust this value based on your preference
        )
        positive_distance_reward = proximity_reward / (
            1 + np.linalg.norm(self.pipette_position - self.goal_position)
        )

        # Goal achievement reward
        goal_achievement_reward = (
            100  # Adjust this value based on your preference
        )
        goal_reward = (
            goal_achievement_reward
            if np.linalg.norm(self.pipette_position - self.goal_position)
            < 0.01
            else 0
        )

        # Combine the rewards
        reward = (
            negative_distance_reward + positive_distance_reward + goal_reward
        )

        # Check if the task has been completed and if the episode should be terminated
        threshold_distance = (
            0.01  # Adjust this threshold based on your environment
        )
        terminated = (
            np.linalg.norm(self.pipette_position - self.goal_position)
            < threshold_distance
            or self.steps >= self.MAX_STEPS
        )

        # Print relevant information for debugging
        # print(f"Reward: {reward}, Terminated: {terminated}, Goal Reward: {goal_reward}")

        return reward, terminated

    def get_position(self):
        # Return the current position of the pipette
        return self.pipette_position

    def close(self):
        self.sim.close()


# Generated by ChatGPT
# Your prompt for generating this code was:
# [pleas add formatting, comments and optimize this code]
