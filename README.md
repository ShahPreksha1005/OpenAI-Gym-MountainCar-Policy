# OpenAI Gym - MountainCar Policy Exploration

## Overview
This repository explores the **MountainCar-v0** environment from OpenAI Gym using a simple momentum-based policy. The project demonstrates environment setup, visualization using IPython, and performance evaluation based on episode rewards.

## Features
- **MountainCar-v0 Environment:** Simulates an underpowered car that must build momentum to reach a flag at the top of a hill.
- **Simple Policy:** Moves left if velocity is negative, otherwise moves right.
- **Frame Visualization:** Captures and displays each frame inline in a Jupyter Notebook.
- **Performance Tracking:** Logs episode rewards and steps taken.

## Installation
To set up the environment, install the necessary libraries:

```bash
pip install gymnasium matplotlib numpy pandas pillow
```

## Implementation Steps
1. **Initialize the Environment**
   ```python
   import gymnasium as gym
   env = gym.make("MountainCar-v0", render_mode="rgb_array")
   ```

2. **Define a Simple Policy**
   ```python
   def simple_policy(state):
       _, velocity = state
       return 2 if velocity >= 0 else 0  # Move right if velocity is positive, else left
   ```

3. **Display Frames in Jupyter Notebook**
   ```python
   from PIL import Image
   from IPython.display import display, clear_output

   def display_frame(env):
       frame = env.render()
       img = Image.fromarray(frame)
       display(img)
       clear_output(wait=True)
   ```

4. **Run Episodes and Track Performance**
   ```python
   num_episodes = 5
   episode_rewards = []
   episode_steps = []

   for episode in range(num_episodes):
       state, _ = env.reset()
       done = False
       total_reward = 0
       step_count = 0

       while not done:
           action = simple_policy(state)
           state, reward, done, truncated, info = env.step(action)
           total_reward += reward
           step_count += 1
           display_frame(env)
           
           if done:
               print(f"Episode {episode + 1} finished in {step_count} steps with total reward {total_reward}")
               episode_rewards.append(total_reward)
               episode_steps.append(step_count)
               break

   env.close()
   ```

## Results
The table below summarizes episode rewards and step counts:

| Episode | Total Reward | Steps Taken |
|---------|-------------|-------------|
| 1       | -116.0      | 116         |
| 2       | -115.0      | 115         |
| 3       | -115.0      | 115         |
| 4       | -122.0      | 122         |
| 5       | -116.0      | 116         |

## Key Insights
- **Policy Effectiveness:** The basic policy is easy to implement but does not consistently reach the goal due to the lack of learning-based adaptation.
- **Visualization:** Displaying frames in Jupyter Notebook allows real-time monitoring of the agent’s behavior.
- **Performance Analysis:** Logging steps and rewards per episode provides insights into the policy's limitations.

## Next Steps
- Implement **Q-learning** for better policy optimization.
- Experiment with **Deep Q Networks (DQN)** for improved performance.

## References
- [OpenAI Gym Documentation](https://www.gymlibrary.dev/)

---
