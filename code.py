import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Provided grid information
random.seed(int(round(time.time() * 1000)), 2)
np.set_printoptions(precision=1)

# Grid setup
rewards = np.zeros((10, 10))
rewards[0, 8] = -500
rewards[8, 6] = -100
rewards[5, 9] = 1000
illegal_positions = [[1, 2], [2, 2], [3, 2], [4, 2], [3, 6], [4, 6], [5, 6], [6, 6], [5, 7], [6, 7]]
terminal_pos = [[5, 9]]

# Q-learning parameters and initialization
Q_matrix_old = np.zeros((100, 4))
Q_matrix_current = Q_matrix_old.copy()
learning_rate = 1
discount = 0.8
epochs = 29

# Helper function to check if a position is a valid starting position
def is_valid_starting_position(position):
    return (position[0] == 0 and position[1] < 3) or (position[0] == 9 and position[1] < 3)

# Lists to store statistics
max_q_values = []  # Maximum Q-value for a state over episodes
episode_numbers = []  # Episode numbers
error_rates = []  # Error rate (difference between new and old Q-values)
actions_made = []  # Actions made for each epoch

# Function to move the robber
def move(curr_position, direction):
    new_pos = []
    if direction == 0:  # Up
        new_pos = [curr_position[0] - 1, curr_position[1]]
    elif direction == 1:  # Left
        new_pos = [curr_position[0], curr_position[1] - 1]
    elif direction == 2:  # Down
        new_pos = [curr_position[0] + 1, curr_position[1]]
    elif direction == 3:  # Right
        new_pos = [curr_position[0], curr_position[1] + 1]

    if new_pos[0] < 0 or new_pos[0] > 9 or new_pos[1] < 0 or new_pos[1] > 9:
        new_pos = curr_position.copy()
    elif new_pos in illegal_positions:
        new_pos = curr_position.copy()

    return new_pos

# Function to find the maximum Q-value for a state
def maxQ(curr_position, Q_matrix):
    return max(Q_matrix[curr_position[0] * 10 + curr_position[1]])

# Function to extract the optimal path from the Q-matrix
def extract_optimal_path(Q_matrix):
    start_position = [0, 0]
    path = [start_position]
    current_position = start_position

    while current_position not in terminal_pos:
        current_q_index = current_position[0] * 10 + current_position[1]
        best_move = np.argmax(Q_matrix[current_q_index])
        next_position = move(current_position, best_move)
        path.append(next_position)
        current_position = next_position

    return path

# Iterate through different starting positions
for row_start in [0, 9]:
    for col_start in range(3):
        if not is_valid_starting_position([row_start, col_start]):
            continue

        for episode in range(epochs):
            position = [row_start, col_start]
            iteration_actions = [0 for i in range(4)]

            start_pos = position.copy()

            while position not in terminal_pos:
                move_direction = random.choice(range(4))

                iteration_actions[move_direction] += 1

                current_state_q_index = position[0] * 10 + position[1]

                position = move(position, move_direction)

                old_Q = Q_matrix_old[current_state_q_index][move_direction]
                new_state_reward = -5 + rewards[position[0]][position[1]]

                Q_matrix_current[current_state_q_index][move_direction] = \
                    old_Q + learning_rate * (new_state_reward + discount * maxQ(position, Q_matrix_old) - old_Q)

            actions_made.append([start_pos, iteration_actions])
            Q_matrix_old = Q_matrix_current.copy()

            # Calculate error rate
            error_rate = np.mean(np.abs(Q_matrix_current - Q_matrix_old))
            error_rates.append(error_rate)

            # Store the maximum Q-value for a state and the corresponding episode number
            max_q_values.append(np.max(Q_matrix_current))
            episode_numbers.append(episode + 1)  # Episode numbers start from 1

# Display the directions moved for each epoch
print("Directions moved for", epochs, "epochs")
print("Directions are [[start position], [up, left, down, right]] for each epoch")
for line in actions_made:
    print(line)

# Plotting for Q matrix
Q_max = np.zeros((10, 10))
Q_max_lower_bound = 0
Q_max_upper_bound = 0

for i in range(10):
    for j in range(10):
        Q_max_value = sum(Q_matrix_current[i * 10 + j]) / 4
        Q_max[i, j] = Q_max_value

        if Q_max_value > Q_max_upper_bound:
            Q_max_upper_bound = Q_max_value
        if Q_max_value < Q_max_upper_bound:
            Q_max_lower_bound = Q_max_value

# Plotting for convergence of Q-values
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(episode_numbers, max_q_values, marker='o')
plt.xlabel('Episodes')
plt.ylabel('Max Q-value')
plt.title('Convergence of Q-values')

# Plotting for number of samples/episodes vs. Maximum Q-value
plt.subplot(2, 2, 2)
plt.plot(episode_numbers, max_q_values, marker='o')
plt.xlabel('Number of Episodes')
plt.ylabel('Max Q-value')
plt.title('Number of Episodes vs. Max Q-value')

# Plotting for error rate vs. episodes
plt.subplot(2, 1, 2)
plt.plot(episode_numbers, error_rates, marker='o', color='r')
plt.xlabel('Episodes')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. Episodes')

# Plotting for number of epochs needed for evolution of error rate
plt.figure(figsize=(8, 6))
plt.plot(error_rates, marker='o', color='b')
plt.xlabel('Number of Epochs')
plt.ylabel('Error Rate')
plt.title('Number of Epochs vs. Error Rate')

# Plotting the grid world with optimal path
optimal_path = extract_optimal_path(Q_matrix_current)
grid_world = np.zeros((10, 10))
for pos in illegal_positions:
    grid_world[pos[0], pos[1]] = -1

for pos in optimal_path:
    grid_world[pos[0], pos[1]] = 1

plt.figure(figsize=(8, 8))
plt.imshow(grid_world, cmap='coolwarm', interpolation='nearest')
plt.title('Grid World with Optimal Path')
plt.colorbar()
plt.show()


plt.figure(figsize=(8, 4))
plt.subplot(122)
plt.plot(error_rates)
plt.title("Evolvement of Error Rate")
plt.xlabel("Epochs")
plt.ylabel("Error Rate")
plt.tight_layout()

plt.show()