import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pygame
import gymnasium as gym
from gymnasium import spaces
import matplotlib.path as mplpath
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as poly
import random
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import shapely.geometry
from rectpack import newPacker
class MovingPolygonsEnv(gym.Env):
    def __init__(self):
        # Define the size of the grid and other parameters
        self.WIDTH, self.HEIGHT = 4000, 4000
        self.Scaled_width = self.WIDTH * 0.2
        self.scaled_height = self.HEIGHT * 0.2
        self.SCREEN_SIZE = (800, 800)
        self.BG_COLOR = (0, 0, 0)
        self.SELECTED_BORDER_COLOR = (255, 0, 0)
        self.MAX_POLYGONS = 20  # Adjust this based on your preference
        self.MAX_POLYGON_POINTS = 14  # Maximum number of points in a polygon
        self.selected_polygon_index = 0
        self.score = 0.0
        # Create the Gym observation space for agent and target locations
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8) 

        # Define the Gym action space (five possible actions)
        self.action_space = spaces.Discrete(7)

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.SCREEN_SIZE)
        pygame.display.set_caption("Moving Polygons")

        # Initialize other variables and create polygons
        self.polygons = []
        self.selected_polygon = None
        self.active_polygon = None
        self.font = pygame.font.Font(None, 36)
        self.score = 0.0

    def reset(self, seed=None):
        # Reset the environment, initialize polygons, and return the initial observation
        self.polygons = self._create_polygons()
        self.initialize_random_positions(self.polygons)
        self.selected_polygon = None
        self.active_polygon = None
        self.score = 0.0
        info={}
        observation = self._get_observation()
        return observation, info

    def step(self, action):
        # Get the currently selected polygon based on the index
        selected_polygon = self.polygons[self.selected_polygon_index]
        selected_polygon.selected = True
        # print(selected_polygon.scaled_points)
        # Execute the specified action on the selected polygon
        if action == 1:
            selected_polygon.move(1,self.polygons)  # Move left
            
        elif action == 2:
            selected_polygon.move(2,self.polygons)  # Move right
            
        elif action == 3:
            selected_polygon.move(3,self.polygons)  # Move up
            
        elif action == 4:
            selected_polygon.move(4,self.polygons)  # Move down
            
        elif action == 5:
            selected_polygon.move(5,self.polygons)  # CW Rotate
            
        elif action == 6:
            selected_polygon.move(6,self.polygons)  # ACW Rotate
            

        
        bounding_box_area = self._calculate_bounding_box_area()
        total_polygon_area = sum(self._calculate_polygon_area(p.scaled_points) for p in self.polygons)
        self.ratio = (total_polygon_area / bounding_box_area) * 100
        self.score = (total_polygon_area / bounding_box_area) * 100 

        observation = self._get_observation()
        reward = (self.score)**3
        done = (self.score == 100)

        # Update the selected polygon index for the next step
        self.selected_polygon_index = (self.selected_polygon_index + 1) % len(self.polygons)
        l=[]
        for p in self.polygons:
            l+=[[(x + p.x, y + p.y) for x, y in p.scaled_points]]

        info = {"points:":l}

        return observation, reward, done, False, info
    def render(self, mode='human'):
        # Visualize the environment
        self.screen.fill(self.BG_COLOR)

        for polygon in self.polygons:
            if polygon.selected:
                pygame.draw.polygon(self.screen, self.SELECTED_BORDER_COLOR,
                                    [(x + polygon.x, y + polygon.y) for x, y in polygon.scaled_points], width=2)
                pygame.draw.polygon(self.screen, polygon.color,
                                    [(x + polygon.x, y + polygon.y) for x, y in polygon.scaled_points])
            else:
                pygame.draw.polygon(self.screen, polygon.color,
                                    [(x + polygon.x, y + polygon.y) for x, y in polygon.scaled_points])

        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()

    def close(self):
        pygame.quit()

    def _create_polygons(self):
        # Create random polygons
        polygons = [
    Polygon([(0, 86), (966, 142), (1983, 0), (2185, 238), (2734, 217), (3000, 767), (2819, 900), (2819, 1360), (3000, 1493), (2734, 2043), (2185, 2022), (1983, 2260), (966, 2118), (0, 2174)], (255, 0, 0)),  # Red
    Polygon([(0, 86), (966, 142), (1983, 0), (2185, 238), (2734, 217), (3000, 767), (2819, 900), (2819, 1360), (3000, 1493), (2734, 2043), (2185, 2022), (1983, 2260), (966, 2118), (0, 2174)], (0, 255, 0)),  # Green
    Polygon([(0, 0), (3034, 0), (3034, 261), (0, 261)], (0, 0, 255)),  # Blue
    Polygon([(0, 0), (3034, 0), (3034, 261), (0, 261)], (255, 255, 0)),  # Yellow
    Polygon([(74, 0), (870, 119), (1666, 0), (1740, 125), (870, 305), (0, 125)], (255, 0, 255)),  # Purple
    Polygon([(74, 0), (870, 119), (1666, 0), (1740, 125), (870, 305), (0, 125)], (0, 255, 255)),  # Cyan
    Polygon([(74, 0), (870, 119), (1666, 0), (1740, 125), (870, 305), (0, 125)], (255, 128, 0)),  # Orange
    Polygon([(74, 0), (870, 119), (1666, 0), (1740, 125), (870, 305), (0, 125)], (128, 0, 255)),  # Magenta
    Polygon([(0, 173), (1761, 0), (2183, 650), (2183, 1010), (1761, 1660), (0, 1487)], (128, 128, 128)),  # Gray
    Polygon([(0, 173), (1761, 0), (2183, 650), (2183, 1010), (1761, 1660), (0, 1487)], (255, 255, 255)),  # White
    Polygon([(0, 173), (1761, 0), (2183, 650), (2183, 1010), (1761, 1660), (0, 1487)], (0, 128, 0)), 
    Polygon([(0, 173), (1761, 0), (2183, 650), (2183, 1010), (1761, 1660), (0, 1487)], (255, 128, 128)),  # Pink
    Polygon([(0, 0), (411, 65), (800, 0), (1189, 65), (1600, 0), (1500, 368), (800, 286), (100, 368)], (128, 255, 128)),  # Light Green
    Polygon([(0, 0), (411, 65), (800, 0), (1189, 65), (1600, 0), (1500, 368), (800, 286), (100, 368)], (255, 0, 128)),  # Pink
    Polygon([(0, 0), (411, 65), (800, 0), (1189, 65), (1600, 0), (1500, 368), (800, 286), (100, 368)], (128, 0, 255)),  # Purple
    Polygon([(0, 0), (411, 65), (800, 0), (1189, 65), (1600, 0), (1500, 368), (800, 286), (100, 368)], (255, 255, 128)),  # Light Yellow
    Polygon([(0, 0), (936, 0), (936, 659), (0, 659)], (0, 255, 255)),  # Light Cyan
    Polygon([(0, 0), (936, 0), (936, 659), (0, 659)], (128, 255, 255)),  # Light Blue
    Polygon([(0, 0), (936, 0), (936, 659), (0, 659)], (255, 128, 255)),  # Light Purple
    Polygon([(0, 0), (936, 0), (936, 659), (0, 659)], (255, 255, 0)),  # Light Yellow
    Polygon([(56, 73), (1066, 143), (1891, 0), (2186, 288), (2573, 241), (2676, 926), (2594, 1366), (0, 1366)], (0, 128, 255)),  # Light Blue
    Polygon([(56, 73), (1066, 143), (1891, 0), (2186, 288), (2573, 241), (2676, 926), (2594, 1366), (0, 1366)], (128, 255, 0)),  # Light Green
    Polygon([(0, 0), (2499, 0), (2705, 387), (2622, 934), (2148, 967), (1920, 1152), (1061, 1059), (0, 1125)], (255, 0, 0))  # Red
]

        # for _ in range(self.MAX_POLYGONS):
        #     points = [(random.randint(0, int(self.Scaled_width)), random.randint(0, int(self.scaled_height))) for _ in
        #               range(self.MAX_POLYGON_POINTS)]
        #     color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #     polygons.append(Polygon(points, color))

        return polygons

    def _pair_overlaps(self):
        # Check for overlaps between pairs of polygons
        c = 0

        for polygon1 in self.polygons:
            for polygon2 in self.polygons:
                if polygon1 != polygon2:
                    if self._overlap(polygon1, polygon2):
                        c += 1

        return c
    
    def rotate_point(self,point, angle, center):
        """Rotate a point by a given angle (in degrees) around a specified center point."""
        x, y = point
        cx, cy = center
        angle_rad = np.deg2rad(angle)
        new_x = cx + (x - cx) * np.cos(angle_rad) - (y - cy) * np.sin(angle_rad)
        new_y = cy + (x - cx) * np.sin(angle_rad) + (y - cy) * np.cos(angle_rad)
        return new_x, new_y

    def rotate_polygon(self, polygon, angle):
        """Rotate all points in a polygon by a given angle (in degrees) around its center."""
        # Calculate the center of the polygon
        x_coords, y_coords = zip(*polygon)
        center = (np.mean(x_coords), np.mean(y_coords))
        
        # Rotate each point around the center
        rotated_polygon = [self.rotate_point(point, angle, center) for point in polygon]
        return rotated_polygon

    def overlap(self,polygon1, polygon2):
        """Check if two polygons overlap."""
        path1 = shapely.geometry.Polygon([(x + polygon1.x , y + polygon1.y) for x, y in polygon1.scaled_points])
        path2 = shapely.geometry.Polygon([(x + polygon2.x, y + polygon2.y) for x, y in polygon2.scaled_points])
        return path1.intersects(path2)
    def check_all_collisions(self,polygons):
        for i in range(len(self.polygons)):
            if self.polygons[i].selected == True:
                p = self.polygons[i]
        for j in range(len(self.polygons)):
                if self.polygons[j] != p:
                    if self.overlap(self.polygons[j], p):
                        return True
        return False
    def _get_observation(self):
        surf = pygame.Surface((64, 64))
        
        # Draw polygons on image
        for polygon in self.polygons:
            pygame.draw.polygon(surf, polygon.color, [(x + polygon.x, y + polygon.y) for x, y in polygon.scaled_points])

        # Convert PyGame surface to NumPy array
        obs = np.flipud(np.rot90(pygame.surfarray.array3d(surf)))

        return obs

    def initialize_random_positions(self,polygons):
        for polygon in polygons:
            while True:
                # Generate random positions within the screen boundaries
                x = random.randint(0, self.Scaled_width - polygon.Scaled_width)
                y = random.randint(0, self.scaled_height - polygon.scaled_height)
                polygon.x = x
                polygon.y = y

                # Check if the generated position overlaps with any existing polygons
                if not self.check_collision_with_existing(polygon, polygons):
                    break

    def check_collision_with_existing(self,polygon, polygons):
        for existing_polygon in polygons:
            if polygon != existing_polygon and self.overlap(polygon, existing_polygon):
                return True
        return False

    def _calculate_bounding_box_area(self):
        # Calculate the bounding box area of all polygons
        min_x = min(p.x for p in self.polygons)
        max_x = max(p.x + p.Scaled_width for p in self.polygons)
        min_y = min(p.y for p in self.polygons)
        max_y = max(p.y + p.scaled_height for p in self.polygons)
        return (max_x - min_x) * (max_y - min_y)

    def _calculate_polygon_area(self, polygon):
        # Calculate the area of a polygon using the shoelace formula
        area = 0.0
        num_points = len(polygon)

        for i in range(num_points):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % num_points]
            area += (x1 * y2 - x2 * y1)

        return abs(area) / 2.0
    


# Define a class for polygons
class Polygon:
    def __init__(self, points, color):
        self.points = points
        self.scaled_points = [(int(x * 0.05), int(y * 0.05)) for x, y in self.points]
        self.color = color
        self.speed = 50
        self.width, self.height = max(x for x, y in points), max(y for x, y in points)
        self.Scaled_width = int(self.width * 0.05)
        self.scaled_height = int(self.height * 0.05)
        # Generate random coordinates within the screen boundaries, avoiding overlap
        self.x = 0
        self.y = 0
        self.selected = False  # Flag to indicate if this polygon is selected


    def move(self, keys, polygons):
        if self.selected:
            
            if keys==1:
                self.x -= self.speed
                if MovingPolygonsEnv().check_all_collisions(polygons):
                    self.x += self.speed  # Revert the movement if a collision occurs

            if keys==2:
                self.x += self.speed
                if MovingPolygonsEnv().check_all_collisions(polygons):
                    self.x -= self.speed

            if keys==3:
                self.y -= self.speed
                if MovingPolygonsEnv().check_all_collisions(polygons):
                    self.y += self.speed

            if keys==4:
                self.y += self.speed
                if MovingPolygonsEnv().check_all_collisions(polygons):
                    self.y -= self.speed

            if keys==5:
                self.scaled_points = MovingPolygonsEnv().rotate_polygon(self.scaled_points, self.speed)
                if MovingPolygonsEnv().check_all_collisions(polygons):
                    self.scaled_points = MovingPolygonsEnv().rotate_polygon(self.scaled_points, -self.speed)
            
            if keys==6:
                self.scaled_points = MovingPolygonsEnv().rotate_polygon(self.scaled_points, -self.speed)
                if MovingPolygonsEnv().check_all_collisions(polygons):
                    self.scaled_points = MovingPolygonsEnv().rotate_polygon(self.scaled_points, +self.speed)
# Create the Gym environment
env = MovingPolygonsEnv()

# Wrap the environment with DummyVecEnv to handle multiple environments if needed
env = DummyVecEnv([lambda: env])

# Create the PPO model
model = PPO("CnnPolicy", env, verbose=1)  # You can adjust hyperparameters here

# Number of training episodes
num_episodes = 3

# Initialize variables to keep track of the best observation and score
best_observation = None
best_score = -float('inf')
best_ratio= -float('inf')
episode_steps = 0 
for episode in range(num_episodes):
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)  # Get the model's action

        # Take the action and observe the environment
        if episode_steps<20000:
            obs, reward, done,info = env.step(action)
            episode_steps += 1  # Increment the step count for this episode
        else:
            done = True

        # You can print or use the results as needed
        print(f"Episode: {episode + 1}, Step: {episode_steps}, Action: {action}, Reward: {reward}")

    current_score = reward

    # Check if the current score is better than the best score so far
    if current_score > best_score:
        points = info
        best_score = current_score
        best_observation = obs.copy()  # Create a copy of the observation to store the best one

    # Train the model at the end of each episode (optional)
    model.learn(total_timesteps=episode_steps)

# Save the trained model
model.save("ppo_polygon_packing")
# Close the environment
env.close()
if best_observation is not None:
    print(points)
    print(f"Best Score: {best_score}")
    print(f"Percentage: {best_ratio}%")