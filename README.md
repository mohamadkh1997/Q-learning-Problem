# Q-learning-Problem


Grid World Parameters:

Grid Size: 10x10

Non-navigable positions:
[1, 2], [2, 2], [3, 2], [4, 2]
[3, 6], [4, 6], [5, 6], [6, 6]
[5, 7], [6, 7]

Prize Position: [5, 9] (Reward: +1000)

Penalty Positions: [0, 8], [8, 6] (Penalty: -500, -100)




Agent Parameters:

Agent Representation: Robber

Starting Positions:
2 or 3 positions from rows 0 and 9 before column 3

Possible Moves: 4 directions (up, down, left, right)

Movement Cost: -5 per move

Probability of Choosing Each Direction: 0.25

Scenario:

The scenario involves the agent (robber) using Q-learning to find the most optimal path to reach the prize (loot) in the grid world. The agent's objective is to maximize the total reward while navigating through the grid and avoiding non-navigable positions and penalties.

