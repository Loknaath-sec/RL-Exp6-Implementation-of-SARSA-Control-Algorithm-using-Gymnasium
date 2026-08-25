# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement
The objective of this experiment is to implement the SARSA (State-Action-Reward-State-Action) control algorithm using the Gymnasium FrozenLake-v1 environment. The agent learns an action-value function (Q-table) through repeated interaction with the environment. It uses an epsilon-greedy policy to balance exploration and exploitation and learns to select suitable actions for reaching the goal while avoiding unsafe


## Software Requirements
Python 3.x
Jupyter Notebook
NumPy
Matplotlib
Gymnasium
FrozenLake-v1 environment
A system capable of running Python programs


## Environment Description
The experiment uses the FrozenLake-v1 environment provided by Gymnasium. The environment represents a grid where the agent starts from a starting state and must reach the goal state while avoiding holes. The agent can move in four directions: Left, Down, Right, and Up. Since is_slippery=True is used, the agent may sometimes move in a different direction than intended. This introduces uncertainty and makes the learning problem more realistic. The SARSA algorithm learns from the actions actually selected by the agent and gradually improves the Q-values through repeated episodes. The custom layout used in this experiment places the starting state in the first row and the goal state in the last row.


## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm
1. Initialize the FrozenLake environment and create a Q-table with zeros for all state-action pairs.
2. Set the learning rate (α), discount factor (γ), exploration rate (ε), and number of training episodes.
3. eset the environment and select the initial action using the epsilon-greedy policy.
4. Take the selected action, observe the reward and next state, and check whether the episode has ended.
5. If the episode is not finished, select the next action using the epsilon-greedy policy.
6. Update the Q-value using the SARSA rule:
Q(s,a) ← Q(s,a) + α[R + γQ(s',a') − Q(s,a)]
7. Repeat the process for all episodes, gradually reduce ε, and finally use the learned Q-table to obtain the state-value function and learned policy.

## Python Program
```py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
```

```py
# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make(
    "FrozenLake-v1",
    desc=[
        "FFSF",   # Starting state is in the first row
        "FFFF",
        "FFFF",
        "FGFF"    # Goal/ending state is in the last row
    ],
    is_slippery=True
)

state_size = env.observation_space.n
action_size = env.action_space.n

print("Number of states:", state_size)
print("Number of actions:", action_size)
```

```py
# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor

epsilon = 1.0        # Initial exploration rate
epsilon_min = 0.05   # Minimum exploration rate
epsilon_decay = 0.9995
```

```py
# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((state_size, action_size))
episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    """
    Select an action using an epsilon-greedy strategy.
    """

    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    return int(np.argmax(Q[state]))
```

```py
# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

for episode in range(num_episodes):

    # Reset environment
    state, info = env.reset()

    # Select first action using epsilon-greedy policy
    action = epsilon_greedy_action(state, epsilon)

    total_reward = 0

    for step in range(max_steps_per_episode):

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_reward += reward

        # -------------------------------------------------
        # If episode is finished
        # -------------------------------------------------
        if done:

            target = reward

            Q[state, action] += alpha * (
                target - Q[state, action]
            )

            break

        # -------------------------------------------------
        # Select next action using current policy
        # -------------------------------------------------

        next_action = epsilon_greedy_action(
            next_state,
            epsilon
        )

        # -------------------------------------------------
        # SARSA Update
        # -------------------------------------------------

        target = reward + gamma * Q[next_state, next_action]

        Q[state, action] += alpha * (
            target - Q[state, action]
        )

        # Move to next state and action
        state = next_state
        action = next_action

    # Store episode reward
    episode_rewards.append(total_reward)

    # Reduce exploration gradually
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# Estimate State Values and Learned Policy
# -------------------------------------------------

state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)

print("Training completed.")
print("Final epsilon:", round(epsilon, 4))
```

```py
# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )


def print_policy(policy):

    # FrozenLake action mapping
    # 0 = Left
    # 1 = Down
    # 2 = Right
    # 3 = Up

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [
            action_symbols[action]
            for action in policy
        ]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)
```

```py
# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(learned_policy)

# Average reward over the last 1000 episodes
average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    average_reward
)
```

```py
# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
print("Name: Loknaath P")
print("Register Number: 212223240080")

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("SARSA Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()
```




























---

## Output

### Final Q-table:





### Estimated State-Value Function:





### Learned Policy:




### Average reward over last 1000 episodes: 



---

## Result
```text
The SARSA algorithm successfully learns an action-value function for the FrozenLake environment. After training, the Q-table contains the learned values for each state-action pair. The maximum Q-value for each state is used to estimate the state-value function, and the action with the highest Q-value is selected to form the learned policy. The average reward over the final episodes can be used to measure the learning performance of the agent.
```

---

## Inference
```text
SARSA learns an effective policy by updating Q-values using the action actually selected in the next state.
The epsilon-greedy policy allows the agent to explore different actions while gradually learning to exploit better actions.
Because the FrozenLake environment is slippery, the agent must learn to handle uncertain movements and avoid risky paths.
With more training episodes, the Q-values become more stable and the learned policy generally improves.
The final Q-table, state-value function, learned policy, and average reward show how well SARSA has learned to reach the goal.
```
---

