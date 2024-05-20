import pygame
import time
import random
import numpy as np

pygame.init()

# Screen dimensions
width = 600
height = 400

# Colors
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

# Snake settings
snake_block = 10
snake_speed = 50  # Increased speed for faster training

# Initialize game window
window = pygame.display.set_mode((width, height))
pygame.display.set_caption('Snake Game')

clock = pygame.time.Clock()

font_style = pygame.font.SysFont(None, 50)
score_font = pygame.font.SysFont(None, 35)

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x1 = width / 2
        self.y1 = height / 2
        self.x1_change = 0
        self.y1_change = 0
        self.snake_List = []
        self.Length_of_snake = 1
        self.foodx = round(random.randrange(0, width - snake_block) / 10.0) * 10.0
        self.foody = round(random.randrange(0, height - snake_block) / 10.0) * 10.0
        self.game_over = False
        self.reward = 0
        self.death_type = None
        self.steps_since_last_food = 0

    def step(self, action):
        if action == 0:  # Left
            self.x1_change = -snake_block
            self.y1_change = 0
        elif action == 1:  # Right
            self.x1_change = snake_block
            self.y1_change = 0
        elif action == 2:  # Up
            self.y1_change = -snake_block
            self.x1_change = 0
        elif action == 3:  # Down
            self.y1_change = snake_block
            self.x1_change = 0

        self.x1 += self.x1_change
        self.y1 += self.y1_change

        self.steps_since_last_food += 1

        if self.x1 >= width or self.x1 < 0 or self.y1 >= height or self.y1 < 0:
            self.game_over = True
            self.reward = -10
            self.death_type = 'wall collision'
        else:
            self.reward = -0.1  # Small negative reward for each step to encourage faster food finding

        snake_Head = [self.x1, self.y1]
        self.snake_List.append(snake_Head)

        if len(self.snake_List) > self.Length_of_snake:
            del self.snake_List[0]

        for x in self.snake_List[:-1]:
            if x == snake_Head:
                self.game_over = True
                self.reward = -20  # Larger penalty for self-collision
                self.death_type = 'self-collision'

        if self.x1 == self.foodx and self.y1 == self.foody:
            self.foodx = round(random.randrange(0, width - snake_block) / 10.0) * 10.0
            self.foody = round(random.randrange(0, height - snake_block) / 10.0) * 10.0
            self.Length_of_snake += 1
            self.reward = 10 + max(0, 100 - self.steps_since_last_food)  # Reward for eating food quickly
            self.steps_since_last_food = 0

        return self.get_state(), self.reward, self.game_over

    def get_state(self):
        state = [
            self.x1_change == -snake_block,  # Moving left
            self.x1_change == snake_block,  # Moving right
            self.y1_change == -snake_block,  # Moving up
            self.y1_change == snake_block,  # Moving down
            self.foodx < self.x1,  # Food is left
            self.foodx > self.x1,  # Food is right
            self.foody < self.y1,  # Food is up
            self.foody > self.y1,  # Food is down
        ]
        return np.array(state, dtype=float)

    def render(self):
        window.fill(white)
        pygame.draw.rect(window, green, [self.foodx, self.foody, snake_block, snake_block])
        for x in self.snake_List:
            pygame.draw.rect(window, black, [x[0], x[1], snake_block, snake_block])
        display_score(self.Length_of_snake - 1)
        pygame.display.update()

def display_score(score):
    value = score_font.render("Your Score: " + str(score), True, black)
    window.blit(value, [0, 0])

def message(msg, color):
    mesg = font_style.render(msg, True, color)
    window.blit(mesg, [width / 6, height / 3])

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((2**state_size, action_size))  # Updated Q-table size
        self.learning_rate = 0.1
        self.discount_rate = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999  # Slower decay for more exploration
        self.epsilon_min = 0.01

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_index = self.state_to_index(state)
        return np.argmax(self.q_table[state_index])

    def update_q_table(self, state, action, reward, next_state):
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        best_next_action = np.argmax(self.q_table[next_state_index])
        target = reward + self.discount_rate * self.q_table[next_state_index][best_next_action]
        self.q_table[state_index][action] = (1 - self.learning_rate) * self.q_table[state_index][action] + self.learning_rate * target

    def state_to_index(self, state):
        binary_state = state.astype(int)
        index = 0
        for i, val in enumerate(binary_state):
            index += val * (2 ** i)
        return index

def train_snake():
    game = SnakeGame()
    agent = QLearningAgent(state_size=8, action_size=4)  # Adjusted state_size
    episodes = 10000
    highest_score = 0

    for e in range(episodes):
        state = game.reset()
        state = game.get_state()
        total_reward = 0

        while True:
            game.render()
            action = agent.get_action(state)
            next_state, reward, done = game.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if done:
                highest_score = max(highest_score, game.Length_of_snake - 1)
                print(f"Episode {e+1}/{episodes}, Score: {game.Length_of_snake-1}, Total Reward: {total_reward}, Highest Score: {highest_score}, Death: {game.death_type}")
                break

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Reduce the frequency of rendering to speed up training
        if e % 100 == 0:
            time.sleep(0.1)

train_snake()
