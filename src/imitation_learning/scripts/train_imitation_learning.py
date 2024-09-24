#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, LaserScan
from gazebo_msgs.msg import EntityState, ModelState, ModelStates, LinkStates
from gazebo_msgs.srv import SetEntityState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty

import math
import threading
import numpy as np
import time
import copy
from os.path import join
from shutil import copyfile # keep track of generations
import random
import torch
from torch import optim
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union
import h5py
import os
import yaml
from tqdm import tqdm

from memory import ReplayMemory
from models import GAILDiscriminator, GMMILDiscriminator, PWILDiscriminator, REDDiscriminator, SoftActor, \
                   RewardRelabeller, TwinCritic, create_target_network, make_gail_input, mix_expert_agent_transitions
from training import adversarial_imitation_update, behavioural_cloning_update, sac_update, target_estimation_update
from utils import cycle, flatten_list_dicts, lineplot

robot_pose = np.array([-1.8, 1.8], float)
axes = np.array([0,0,0], float)
lidar_data = np.zeros(20)

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class GazeboEnv(Node):

    def __init__(self, absorbing: bool, load_data: bool=False):
        super().__init__('env')

        self.absorbing = absorbing

        if load_data:
            # Load dataset before (potentially) adjusting observation_space (fails assertion check otherwise)
            self.dataset = self.get_dataset()

        #if absorbing:
        #    # Append absorbing indicator bit to state dimension (assumes 1D state space)
        #    self.observation_space = Box(low=np.concatenate([self.env.observation_space.low, np.zeros(1)]),
        #                                 high=np.concatenate([self.env.observation_space.high, np.ones(1)]))

        self.seed = 0
        self.wheel_vel1 = np.array([0,0,0,0], float)
        self.L = 0.125 # distance from the robot center to the wheel
        self.Rw = 0.03 # Radius ot the wheel
        
        self.set_state = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_world = self.create_client(Empty, "/reset_world")
        #self.reset_simulation = self.create_client(Empty, "/reset_simulation")
        self.req = Empty.Request

        self.publisher_robot_vel1 = self.create_publisher(Float64MultiArray, '/robot_1/forward_velocity_controller/commands', 10)
        
        self.set_box1_state = EntityState()
        self.set_box1_state.name = "box1"
        self.set_box1_state.pose.position.x = 0.0
        self.set_box1_state.pose.position.y = 0.0
        self.set_box1_state.pose.position.z = 0.1
        self.set_box1_state.pose.orientation.x = 0.0
        self.set_box1_state.pose.orientation.y = 0.0
        self.set_box1_state.pose.orientation.z = 0.0
        self.set_box1_state.pose.orientation.w = 1.0
        self.box1_state = SetEntityState.Request()

        self.set_box2_state = EntityState()
        self.set_box2_state.name = "box2"
        self.set_box2_state.pose.position.x = 0.0
        self.set_box2_state.pose.position.y = 0.0
        self.set_box2_state.pose.position.z = 0.1
        self.set_box2_state.pose.orientation.x = 0.0
        self.set_box2_state.pose.orientation.y = 0.0
        self.set_box2_state.pose.orientation.z = 0.0
        self.set_box2_state.pose.orientation.w = 1.0
        self.box2_state = SetEntityState.Request()

        self.set_box3_state = EntityState()
        self.set_box3_state.name = "box3"
        self.set_box3_state.pose.position.x = 0.0
        self.set_box3_state.pose.position.y = 0.0
        self.set_box3_state.pose.position.z = 0.1
        self.set_box3_state.pose.orientation.x = 0.0
        self.set_box3_state.pose.orientation.y = 0.0
        self.set_box3_state.pose.orientation.z = 0.0
        self.set_box3_state.pose.orientation.w = 1.0
        self.box3_state = SetEntityState.Request()

        self.set_box4_state = EntityState()
        self.set_box4_state.name = "box4"
        self.set_box4_state.pose.position.x = 0.0
        self.set_box4_state.pose.position.y = 0.0
        self.set_box4_state.pose.position.z = 0.1
        self.set_box4_state.pose.orientation.x = 0.0
        self.set_box4_state.pose.orientation.y = 0.0
        self.set_box4_state.pose.orientation.z = 0.0
        self.set_box4_state.pose.orientation.w = 1.0
        self.box4_state = SetEntityState.Request()

        self.set_box5_state = EntityState()
        self.set_box5_state.name = "box5"
        self.set_box5_state.pose.position.x = 0.0
        self.set_box5_state.pose.position.y = 0.0
        self.set_box5_state.pose.position.z = 0.1
        self.set_box5_state.pose.orientation.x = 0.0
        self.set_box5_state.pose.orientation.y = 0.0
        self.set_box5_state.pose.orientation.z = 0.0
        self.set_box5_state.pose.orientation.w = 1.0
        self.box5_state = SetEntityState.Request()

        self.set_box6_state = EntityState()
        self.set_box6_state.name = "box6"
        self.set_box6_state.pose.position.x = 0.0
        self.set_box6_state.pose.position.y = 0.0
        self.set_box6_state.pose.position.z = 0.1
        self.set_box6_state.pose.orientation.x = 0.0
        self.set_box6_state.pose.orientation.y = 0.0
        self.set_box6_state.pose.orientation.z = 0.0
        self.set_box6_state.pose.orientation.w = 1.0
        self.box6_state = SetEntityState.Request()

        self.set_box7_state = EntityState()
        self.set_box7_state.name = "box7"
        self.set_box7_state.pose.position.x = 0.0
        self.set_box7_state.pose.position.y = 0.0
        self.set_box7_state.pose.position.z = 0.1
        self.set_box7_state.pose.orientation.x = 0.0
        self.set_box7_state.pose.orientation.y = 0.0
        self.set_box7_state.pose.orientation.z = 0.0
        self.set_box7_state.pose.orientation.w = 1.0
        self.box7_state = SetEntityState.Request()

        self.set_box8_state = EntityState()
        self.set_box8_state.name = "box8"
        self.set_box8_state.pose.position.x = 0.0
        self.set_box8_state.pose.position.y = 0.0
        self.set_box8_state.pose.position.z = 0.1
        self.set_box8_state.pose.orientation.x = 0.0
        self.set_box8_state.pose.orientation.y = 0.0
        self.set_box8_state.pose.orientation.z = 0.0
        self.set_box8_state.pose.orientation.w = 1.0
        self.box8_state = SetEntityState.Request()

        self.set_box9_state = EntityState()
        self.set_box9_state.name = "box9"
        self.set_box9_state.pose.position.x = 0.0
        self.set_box9_state.pose.position.y = 0.0
        self.set_box9_state.pose.position.z = 0.1
        self.set_box9_state.pose.orientation.x = 0.0
        self.set_box9_state.pose.orientation.y = 0.0
        self.set_box9_state.pose.orientation.z = 0.0
        self.set_box9_state.pose.orientation.w = 1.0
        self.box9_state = SetEntityState.Request()

        self.set_box10_state = EntityState()
        self.set_box10_state.name = "box10"
        self.set_box10_state.pose.position.x = 0.0
        self.set_box10_state.pose.position.y = 0.0
        self.set_box10_state.pose.position.z = 0.1
        self.set_box10_state.pose.orientation.x = 0.0
        self.set_box10_state.pose.orientation.y = 0.0
        self.set_box10_state.pose.orientation.z = 0.0
        self.set_box10_state.pose.orientation.w = 1.0
        self.box10_state = SetEntityState.Request()

        self.set_box11_state = EntityState()
        self.set_box11_state.name = "box11"
        self.set_box11_state.pose.position.x = 0.0
        self.set_box11_state.pose.position.y = 0.0
        self.set_box11_state.pose.position.z = 0.1
        self.set_box11_state.pose.orientation.x = 0.0
        self.set_box11_state.pose.orientation.y = 0.0
        self.set_box11_state.pose.orientation.z = 0.0
        self.set_box11_state.pose.orientation.w = 1.0
        self.box11_state = SetEntityState.Request()

        self.set_box12_state = EntityState()
        self.set_box12_state.name = "box12"
        self.set_box12_state.pose.position.x = 0.0
        self.set_box12_state.pose.position.y = 0.0
        self.set_box12_state.pose.position.z = 0.1
        self.set_box12_state.pose.orientation.x = 0.0
        self.set_box12_state.pose.orientation.y = 0.0
        self.set_box12_state.pose.orientation.z = 0.0
        self.set_box12_state.pose.orientation.w = 1.0
        self.box12_state = SetEntityState.Request()

        self.set_box13_state = EntityState()
        self.set_box13_state.name = "box13"
        self.set_box13_state.pose.position.x = 0.0
        self.set_box13_state.pose.position.y = 0.0
        self.set_box13_state.pose.position.z = 0.1
        self.set_box13_state.pose.orientation.x = 0.0
        self.set_box13_state.pose.orientation.y = 0.0
        self.set_box13_state.pose.orientation.z = 0.0
        self.set_box13_state.pose.orientation.w = 1.0
        self.box13_state = SetEntityState.Request()

        self.set_box14_state = EntityState()
        self.set_box14_state.name = "box14"
        self.set_box14_state.pose.position.x = 0.0
        self.set_box14_state.pose.position.y = 0.0
        self.set_box14_state.pose.position.z = 0.1
        self.set_box14_state.pose.orientation.x = 0.0
        self.set_box14_state.pose.orientation.y = 0.0
        self.set_box14_state.pose.orientation.z = 0.0
        self.set_box14_state.pose.orientation.w = 1.0
        self.box14_state = SetEntityState.Request()

        #to move head_link to initial position
        self.set_robot_1_state = EntityState()
        self.set_robot_1_state.name = "robot_1"
        self.set_robot_1_state.pose.position.x = -1.8
        self.set_robot_1_state.pose.position.y = 1.8
        self.set_robot_1_state.pose.position.z = 0.15
        self.set_robot_1_state.pose.orientation.x = 0.0
        self.set_robot_1_state.pose.orientation.y = 0.0
        self.set_robot_1_state.pose.orientation.z = 0.0
        self.set_robot_1_state.pose.orientation.w = 1.0
        self.robot_1_state = SetEntityState.Request()                

        self.t = 0
        self.t_limit = 6000

        #self.obs_robot1 = np.array([0, 0], float)
        #self.done = False
        #self.actions = np.array([0,0,0], float)

        self.TIME_DELTA = 0.2
        self.timeouts = False
        self.obs = np.zeros(22)
        self.next_obs = np.zeros(22)
        self.state_reset = np.zeros(22)
        self.goal_x = 1.8
        self.goal_y = -1.8

    def step(self, action, step, max_episode_steps):
        global lidar_data

        #self.done = False
        #self.actions[:] = axes[:]
        #obs = copy.copy(lidar_data)
        action = action[0].to('cpu').detach().numpy().copy()
        #self.get_logger().info(f"action:{action}")

        self.wheel_vel1[0] = (action[0]*math.sin(math.pi/4            ) + action[1]*math.cos(math.pi/4            ) + self.L*action[2])/self.Rw
        self.wheel_vel1[1] = (action[0]*math.sin(math.pi/4 + math.pi/2) + action[1]*math.cos(math.pi/4 + math.pi/2) + self.L*action[2])/self.Rw
        self.wheel_vel1[2] = (action[0]*math.sin(math.pi/4 - math.pi)   + action[1]*math.cos(math.pi/4 - math.pi)   + self.L*action[2])/self.Rw
        self.wheel_vel1[3] = (action[0]*math.sin(math.pi/4 - math.pi/2) + action[1]*math.cos(math.pi/4 - math.pi/2) + self.L*action[2])/self.Rw


        #publish robot1 commands
        #gz_env.publisher_vel1.publish(gz_env.vel_msg1)
        array_forPublish1_vel = Float64MultiArray(data=self.wheel_vel1)  
        self.publisher_robot_vel1.publish(array_forPublish1_vel)

        #'''
        while not gz_env.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.unpause.call_async(Empty.Request())
        except:
            self.get_logger().info("/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(self.TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            self.get_logger().info("/gazebo/pause_physics service call failed")
        #'''

        self.next_obs[:20] = copy.copy(lidar_data)
        self.next_obs[20] = robot_pose[0] - self.goal_x
        self.next_obs[21] = robot_pose[1] - self.goal_y
        dist = math.sqrt((robot_pose[0] - self.goal_x)**2 + (robot_pose[1] - self.goal_y)**2)
        reward = 1/dist

        if(dist <= 0.35):
            done = True
            reward = 10
        elif(np.amin(self.next_obs[:20]) < 0.25):
            reward = -1
            done = False
        else:
            done = False

        # time out
        if step >= max_episode_steps:
            self.get_logger().info("time out")
            done = True

        next_obs_re = torch.tensor(self.next_obs, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state
        if self.absorbing:
            # Add absorbing indicator (zero) to state (absorbing state rewriting done in replay memory)
            next_obs_re = torch.cat([next_obs_re, torch.zeros(next_obs_re.size(0), 1)], dim=1) 

        return next_obs_re, reward, done #next_state, reward, terminal

    def reset(self):
        global axes, lidar_data
        #gz_env.get_logger().info('RESET!')
        '''
        while not self.reset_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            #self.get_logger().info('Resetting the world')
            self.reset_world.call_async(Empty.Request())
        except:
            import traceback
            traceback.print_exc()
        '''
            
        #self.robot_1_state = SetEntityState.Request()
        '''
        self.robot_1_state._state = self.set_robot_1_state
        while not self.set_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.set_state.call_async(self.robot_1_state)
        except rclpy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")
        '''

        rng = np.random.default_rng(self.seed)
        self.seed += 1
        boxes_pos = []
        for i in range(7):
            numbers = rng.choice(9, size=2, replace=False)
            boxes_pos.append(numbers)

        for j in range(7):
            if j==0:
                print(f"boxes0:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box1_state.pose.position.x = -0.6
                    self.set_box1_state.pose.position.y = 2.4
                elif(boxes_pos[j][0] == 1):
                    self.set_box1_state.pose.position.x = 0.0
                    self.set_box1_state.pose.position.y = 2.4
                elif(boxes_pos[j][0] == 2):
                    self.set_box1_state.pose.position.x = 0.6
                    self.set_box1_state.pose.position.y = 2.4
                elif(boxes_pos[j][0] == 3):
                    self.set_box1_state.pose.position.x = -0.6
                    self.set_box1_state.pose.position.y = 1.8
                elif(boxes_pos[j][0] == 4):
                    self.set_box1_state.pose.position.x = 0.0
                    self.set_box1_state.pose.position.y = 1.8
                elif(boxes_pos[j][0] == 5):
                    self.set_box1_state.pose.position.x = 0.6
                    self.set_box1_state.pose.position.y = 1.8
                elif(boxes_pos[j][0] == 6):
                    self.set_box1_state.pose.position.x = -0.6
                    self.set_box1_state.pose.position.y = 1.2
                elif(boxes_pos[j][0] == 7):
                    self.set_box1_state.pose.position.x = 0.0
                    self.set_box1_state.pose.position.y = 1.2
                elif(boxes_pos[j][0] == 8):
                    self.set_box1_state.pose.position.x = 0.6
                    self.set_box1_state.pose.position.y = 1.2

                if(boxes_pos[j][1] == 0):
                    self.set_box2_state.pose.position.x = -0.6
                    self.set_box2_state.pose.position.y = 2.4
                elif(boxes_pos[j][1] == 1):
                    self.set_box2_state.pose.position.x = 0.0
                    self.set_box2_state.pose.position.y = 2.4
                elif(boxes_pos[j][1] == 2):
                    self.set_box2_state.pose.position.x = 0.6
                    self.set_box2_state.pose.position.y = 2.4
                elif(boxes_pos[j][1] == 3):
                    self.set_box2_state.pose.position.x = -0.6
                    self.set_box2_state.pose.position.y = 1.8
                elif(boxes_pos[j][1] == 4):
                    self.set_box2_state.pose.position.x = 0.0
                    self.set_box2_state.pose.position.y = 1.8
                elif(boxes_pos[j][1] == 5):
                    self.set_box2_state.pose.position.x = 0.6
                    self.set_box2_state.pose.position.y = 1.8
                elif(boxes_pos[j][1] == 6):
                    self.set_box2_state.pose.position.x = -0.6
                    self.set_box2_state.pose.position.y = 1.2
                elif(boxes_pos[j][1] == 7):
                    self.set_box2_state.pose.position.x = 0.0
                    self.set_box2_state.pose.position.y = 1.2
                elif(boxes_pos[j][1] == 8):
                    self.set_box2_state.pose.position.x = 0.6
                    self.set_box2_state.pose.position.y = 1.2

            if j==1:
                print(f"boxes2:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box3_state.pose.position.x = 1.2
                    self.set_box3_state.pose.position.y = 2.4
                elif(boxes_pos[j][0] == 1):
                    self.set_box3_state.pose.position.x = 1.8
                    self.set_box3_state.pose.position.y = 2.4
                elif(boxes_pos[j][0] == 2):
                    self.set_box3_state.pose.position.x = 2.4
                    self.set_box3_state.pose.position.y = 2.4
                elif(boxes_pos[j][0] == 3):
                    self.set_box3_state.pose.position.x = 1.2
                    self.set_box3_state.pose.position.y = 1.8
                elif(boxes_pos[j][0] == 4):
                    self.set_box3_state.pose.position.x = 1.8
                    self.set_box3_state.pose.position.y = 1.8
                elif(boxes_pos[j][0] == 5):
                    self.set_box3_state.pose.position.x = 2.4
                    self.set_box3_state.pose.position.y = 1.8
                elif(boxes_pos[j][0] == 6):
                    self.set_box3_state.pose.position.x = 1.2
                    self.set_box3_state.pose.position.y = 1.2
                elif(boxes_pos[j][0] == 7):
                    self.set_box3_state.pose.position.x = 1.8
                    self.set_box3_state.pose.position.y = 1.2
                elif(boxes_pos[j][0] == 8):
                    self.set_box3_state.pose.position.x = 2.4
                    self.set_box3_state.pose.position.y = 1.2

                if(boxes_pos[j][1] == 0):
                    self.set_box4_state.pose.position.x = 1.2
                    self.set_box4_state.pose.position.y = 2.4
                elif(boxes_pos[j][1] == 1):
                    self.set_box4_state.pose.position.x = 1.8
                    self.set_box4_state.pose.position.y = 2.4
                elif(boxes_pos[j][1] == 2):
                    self.set_box4_state.pose.position.x = 2.4
                    self.set_box4_state.pose.position.y = 2.4
                elif(boxes_pos[j][1] == 3):
                    self.set_box4_state.pose.position.x = 1.2
                    self.set_box4_state.pose.position.y = 1.8
                elif(boxes_pos[j][1] == 4):
                    self.set_box4_state.pose.position.x = 1.8
                    self.set_box4_state.pose.position.y = 1.8
                elif(boxes_pos[j][1] == 5):
                    self.set_box4_state.pose.position.x = 2.4
                    self.set_box4_state.pose.position.y = 1.8
                elif(boxes_pos[j][1] == 6):
                    self.set_box4_state.pose.position.x = 1.2
                    self.set_box4_state.pose.position.y = 1.2
                elif(boxes_pos[j][1] == 7):
                    self.set_box4_state.pose.position.x = 1.8
                    self.set_box4_state.pose.position.y = 1.2
                elif(boxes_pos[j][1] == 8):
                    self.set_box4_state.pose.position.x = 2.4
                    self.set_box4_state.pose.position.y = 1.2

            if j==2:
                print(f"boxes3:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box5_state.pose.position.x = -2.4
                    self.set_box5_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 1):
                    self.set_box5_state.pose.position.x = -1.8
                    self.set_box5_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 2):
                    self.set_box5_state.pose.position.x = -1.2
                    self.set_box5_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 3):
                    self.set_box5_state.pose.position.x = -2.4
                    self.set_box5_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 4):
                    self.set_box5_state.pose.position.x = -1.8
                    self.set_box5_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 5):
                    self.set_box5_state.pose.position.x = -1.2
                    self.set_box5_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 6):
                    self.set_box5_state.pose.position.x = -2.4
                    self.set_box5_state.pose.position.y = -0.6
                elif(boxes_pos[j][0] == 7):
                    self.set_box5_state.pose.position.x = -1.8
                    self.set_box5_state.pose.position.y = -0.6
                elif(boxes_pos[j][0] == 8):
                    self.set_box5_state.pose.position.x = -1.2
                    self.set_box5_state.pose.position.y = -0.6

                if(boxes_pos[j][1] == 0):
                    self.set_box6_state.pose.position.x = -2.4
                    self.set_box6_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 1):
                    self.set_box6_state.pose.position.x = -1.8
                    self.set_box6_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 2):
                    self.set_box6_state.pose.position.x = -1.2
                    self.set_box6_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 3):
                    self.set_box6_state.pose.position.x = -2.4
                    self.set_box6_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 4):
                    self.set_box6_state.pose.position.x = -1.8
                    self.set_box6_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 5):
                    self.set_box6_state.pose.position.x = -1.2
                    self.set_box6_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 6):
                    self.set_box6_state.pose.position.x = -2.4
                    self.set_box6_state.pose.position.y = -0.6
                elif(boxes_pos[j][1] == 7):
                    self.set_box6_state.pose.position.x = -1.8
                    self.set_box6_state.pose.position.y = -0.6
                elif(boxes_pos[j][1] == 8):
                    self.set_box6_state.pose.position.x = -1.2
                    self.set_box6_state.pose.position.y = -0.6

            if j==3:
                print(f"boxes4:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box7_state.pose.position.x = -0.6
                    self.set_box7_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 1):
                    self.set_box7_state.pose.position.x = 0.0
                    self.set_box7_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 2):
                    self.set_box7_state.pose.position.x = 0.6
                    self.set_box7_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 3):
                    self.set_box7_state.pose.position.x = -0.6
                    self.set_box7_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 4):
                    self.set_box7_state.pose.position.x = 0.0
                    self.set_box7_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 5):
                    self.set_box7_state.pose.position.x = 0.6
                    self.set_box7_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 6):
                    self.set_box7_state.pose.position.x = -0.6
                    self.set_box7_state.pose.position.y = -0.6
                elif(boxes_pos[j][0] == 7):
                    self.set_box7_state.pose.position.x = 0.0
                    self.set_box7_state.pose.position.y = -0.6
                elif(boxes_pos[j][0] == 8):
                    self.set_box7_state.pose.position.x = 0.6
                    self.set_box7_state.pose.position.y = -0.6

                if(boxes_pos[j][1] == 0):
                    self.set_box8_state.pose.position.x = -0.6
                    self.set_box8_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 1):
                    self.set_box8_state.pose.position.x = 0.0
                    self.set_box8_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 2):
                    self.set_box8_state.pose.position.x = 0.6
                    self.set_box8_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 3):
                    self.set_box8_state.pose.position.x = -0.6
                    self.set_box8_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 4):
                    self.set_box8_state.pose.position.x = 0.0
                    self.set_box8_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 5):
                    self.set_box8_state.pose.position.x = 0.6
                    self.set_box8_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 6):
                    self.set_box8_state.pose.position.x = -0.6
                    self.set_box8_state.pose.position.y = -0.6
                elif(boxes_pos[j][1] == 7):
                    self.set_box8_state.pose.position.x = 0.0
                    self.set_box8_state.pose.position.y = -0.6
                elif(boxes_pos[j][1] == 8):
                    self.set_box8_state.pose.position.x = 0.6
                    self.set_box8_state.pose.position.y = -0.6

            if j==4:
                print(f"boxes5:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box9_state.pose.position.x = 2.4
                    self.set_box9_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 1):
                    self.set_box9_state.pose.position.x = 1.8
                    self.set_box9_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 2):
                    self.set_box9_state.pose.position.x = 1.2
                    self.set_box9_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 3):
                    self.set_box9_state.pose.position.x = 2.4
                    self.set_box9_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 4):
                    self.set_box9_state.pose.position.x = 1.8
                    self.set_box9_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 5):
                    self.set_box9_state.pose.position.x = 1.2
                    self.set_box9_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 6):
                    self.set_box9_state.pose.position.x = 2.4
                    self.set_box9_state.pose.position.y = -0.6
                elif(boxes_pos[j][0] == 7):
                    self.set_box9_state.pose.position.x = 1.8
                    self.set_box9_state.pose.position.y = -0.6
                elif(boxes_pos[j][0] == 8):
                    self.set_box9_state.pose.position.x = 1.2
                    self.set_box9_state.pose.position.y = -0.6

                if(boxes_pos[j][1] == 0):
                    self.set_box10_state.pose.position.x = 2.4
                    self.set_box10_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 1):
                    self.set_box10_state.pose.position.x = 1.8
                    self.set_box10_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 2):
                    self.set_box10_state.pose.position.x = 1.2
                    self.set_box10_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 3):
                    self.set_box10_state.pose.position.x = 2.4
                    self.set_box10_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 4):
                    self.set_box10_state.pose.position.x = 1.8
                    self.set_box10_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 5):
                    self.set_box10_state.pose.position.x = 1.2
                    self.set_box10_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 6):
                    self.set_box10_state.pose.position.x = 2.4
                    self.set_box10_state.pose.position.y = -0.6
                elif(boxes_pos[j][1] == 7):
                    self.set_box10_state.pose.position.x = 1.8
                    self.set_box10_state.pose.position.y = -0.6
                elif(boxes_pos[j][1] == 8):
                    self.set_box10_state.pose.position.x = 1.2
                    self.set_box10_state.pose.position.y = -0.6

            if j==5:
                print(f"boxes6:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box11_state.pose.position.x = -2.4
                    self.set_box11_state.pose.position.y = -1.2
                elif(boxes_pos[j][0] == 1):
                    self.set_box11_state.pose.position.x = -1.8
                    self.set_box11_state.pose.position.y = -1.2
                elif(boxes_pos[j][0] == 2):
                    self.set_box11_state.pose.position.x = -1.2
                    self.set_box11_state.pose.position.y = -1.2
                elif(boxes_pos[j][0] == 3):
                    self.set_box11_state.pose.position.x = -2.4
                    self.set_box11_state.pose.position.y = -1.8
                elif(boxes_pos[j][0] == 4):
                    self.set_box11_state.pose.position.x = -1.8
                    self.set_box11_state.pose.position.y = -1.8
                elif(boxes_pos[j][0] == 5):
                    self.set_box11_state.pose.position.x = -1.2
                    self.set_box11_state.pose.position.y = -1.8
                elif(boxes_pos[j][0] == 6):
                    self.set_box11_state.pose.position.x = -2.4
                    self.set_box11_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 7):
                    self.set_box11_state.pose.position.x = -1.8
                    self.set_box11_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 8):
                    self.set_box11_state.pose.position.x = -1.2
                    self.set_box11_state.pose.position.y = -2.4

                if(boxes_pos[j][1] == 0):
                    self.set_box12_state.pose.position.x = -2.4
                    self.set_box12_state.pose.position.y = -1.2
                elif(boxes_pos[j][1] == 1):
                    self.set_box12_state.pose.position.x = -1.8
                    self.set_box12_state.pose.position.y = -1.2
                elif(boxes_pos[j][1] == 2):
                    self.set_box12_state.pose.position.x = -1.2
                    self.set_box12_state.pose.position.y = -1.2
                elif(boxes_pos[j][1] == 3):
                    self.set_box12_state.pose.position.x = -2.4
                    self.set_box12_state.pose.position.y = -1.8
                elif(boxes_pos[j][1] == 4):
                    self.set_box12_state.pose.position.x = -1.8
                    self.set_box12_state.pose.position.y = -1.8
                elif(boxes_pos[j][1] == 5):
                    self.set_box12_state.pose.position.x = -1.2
                    self.set_box12_state.pose.position.y = -1.8
                elif(boxes_pos[j][1] == 6):
                    self.set_box12_state.pose.position.x = -2.4
                    self.set_box12_state.pose.position.y = -2.4
                elif(boxes_pos[j][1] == 7):
                    self.set_box12_state.pose.position.x = -1.8
                    self.set_box12_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 8):
                    self.set_box12_state.pose.position.x = -1.2
                    self.set_box12_state.pose.position.y = -2.4

            if j==6:
                print(f"boxes7:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box13_state.pose.position.x = -0.6
                    self.set_box13_state.pose.position.y = -1.2
                elif(boxes_pos[j][0] == 1):
                    self.set_box13_state.pose.position.x = 0.0
                    self.set_box13_state.pose.position.y = -1.2
                elif(boxes_pos[j][0] == 2):
                    self.set_box13_state.pose.position.x = 0.6
                    self.set_box13_state.pose.position.y = -1.2
                elif(boxes_pos[j][0] == 3):
                    self.set_box13_state.pose.position.x = -0.6
                    self.set_box13_state.pose.position.y = -1.8
                elif(boxes_pos[j][0] == 4):
                    self.set_box13_state.pose.position.x = 0.0
                    self.set_box13_state.pose.position.y = -1.8
                elif(boxes_pos[j][0] == 5):
                    self.set_box13_state.pose.position.x = 0.6
                    self.set_box13_state.pose.position.y = -1.8
                elif(boxes_pos[j][0] == 6):
                    self.set_box13_state.pose.position.x = -0.6
                    self.set_box13_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 7):
                    self.set_box13_state.pose.position.x = 0.0
                    self.set_box13_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 8):
                    self.set_box13_state.pose.position.x = 0.6
                    self.set_box13_state.pose.position.y = -2.4

                if(boxes_pos[j][1] == 0):
                    self.set_box14_state.pose.position.x = -0.6
                    self.set_box14_state.pose.position.y = -1.2
                elif(boxes_pos[j][1] == 1):
                    self.set_box14_state.pose.position.x = 0.0
                    self.set_box14_state.pose.position.y = -1.2
                elif(boxes_pos[j][1] == 2):
                    self.set_box14_state.pose.position.x = 0.6
                    self.set_box14_state.pose.position.y = -1.2
                elif(boxes_pos[j][1] == 3):
                    self.set_box14_state.pose.position.x = -0.6
                    self.set_box14_state.pose.position.y = -1.8
                elif(boxes_pos[j][1] == 4):
                    self.set_box14_state.pose.position.x = 0.0
                    self.set_box14_state.pose.position.y = -1.8
                elif(boxes_pos[j][1] == 5):
                    self.set_box14_state.pose.position.x = 0.6
                    self.set_box14_state.pose.position.y = -1.8
                elif(boxes_pos[j][1] == 6):
                    self.set_box14_state.pose.position.x = -0.6
                    self.set_box14_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 7):
                    self.set_box14_state.pose.position.x = 0.0
                    self.set_box14_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 8):
                    self.set_box14_state.pose.position.x = 0.6
                    self.set_box14_state.pose.position.y = -2.4

        # replace models
        self.box1_state._state = self.set_box1_state
        self.box2_state._state = self.set_box2_state
        self.box3_state._state = self.set_box3_state
        self.box4_state._state = self.set_box4_state
        self.box5_state._state = self.set_box5_state
        self.box6_state._state = self.set_box6_state
        self.box7_state._state = self.set_box7_state
        self.box8_state._state = self.set_box8_state
        self.box9_state._state = self.set_box9_state
        self.box10_state._state = self.set_box10_state
        self.box11_state._state = self.set_box11_state
        self.box12_state._state = self.set_box12_state
        self.box13_state._state = self.set_box13_state
        self.box14_state._state = self.set_box14_state
        self.robot_1_state._state = self.set_robot_1_state
        #'''
        while not self.set_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')
        try:
            self.get_logger().info('reset positions')
            self.set_state.call_async(self.box1_state)
            self.set_state.call_async(self.box2_state)
            self.set_state.call_async(self.box3_state)
            self.set_state.call_async(self.box4_state)
            self.set_state.call_async(self.box5_state)
            self.set_state.call_async(self.box6_state)
            self.set_state.call_async(self.box7_state)
            self.set_state.call_async(self.box8_state)
            self.set_state.call_async(self.box9_state)
            self.set_state.call_async(self.box10_state)
            self.set_state.call_async(self.box11_state)
            self.set_state.call_async(self.box12_state)
            self.set_state.call_async(self.box13_state)
            self.set_state.call_async(self.box14_state)
            self.set_state.call_async(self.robot_1_state)
        except:
            import traceback
            traceback.print_exc()

        #self.obs_robot1[0] = -1.8
        #self.obs_robot1[1] = 1.8

        #state = copy.copy(lidar_data)
        self.state_reset [:20] = copy.copy(lidar_data)
        self.state_reset [20] = robot_pose[0] - self.goal_x
        self.state_reset [21] = robot_pose[1] - self.goal_y

        state  = torch.tensor(self.state_reset , dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state
        if self.absorbing:
            state  = torch.cat([state, torch.zeros(state.size(0), 1)], dim=1)  # Add absorbing indicator (zero) to state

        #/7.465 #7.465 is the longest distance from the robot to a wall
        #gz_env.get_logger().info(f"robot1: {self.obs_robot1}")

        return state 
    
    def get_dataset(self, trajectories: int=0, subsample: int=1) -> ReplayMemory:
        # Extract data
        f = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/data/training_data.hdf5'
        expert_data = h5py.File(f,'r')
        states = torch.as_tensor(expert_data['observations'], dtype=torch.float32)
        actions = torch.as_tensor(expert_data['actions'], dtype=torch.float32)
        next_states = torch.as_tensor(expert_data['next_observations'], dtype=torch.float32)
        terminals = torch.as_tensor(expert_data['terminals'], dtype=torch.float32)
        timeouts = torch.as_tensor(expert_data['timeouts'], dtype=torch.float32)
        state_size = states.size(1)
        action_size = actions.size(1)
        # Split into separate trajectories
        states_list = []
        actions_list = []
        next_states_list = []
        terminals_list = []
        weights_list = []
        timeouts_list = []
        terminal_idxs = terminals.nonzero().flatten()
        timeout_idxs = timeouts.nonzero().flatten()
        ep_end_idxs = torch.sort(torch.cat([torch.tensor([-1]), terminal_idxs, timeout_idxs], dim=0))[0]
        for i in range(len(ep_end_idxs) - 1):
          states_list.append(states[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])
          actions_list.append(actions[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])
          next_states_list.append(next_states[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])
          terminals_list.append(terminals[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])  # Only store true terminations; timeouts should not be treated as such
          timeouts_list.append(timeouts[ep_end_idxs[i] + 1:ep_end_idxs[i + 1] + 1])  # Store if episode terminated due to timeout
          weights_list.append(torch.ones_like(terminals_list[-1]))  # Add an importance weight of 1 to every transition
        # Pick number of trajectories
        if trajectories > 0:
          states_list = states_list[:trajectories]
          actions_list = actions_list[:trajectories]
          next_states_list = next_states_list[:trajectories]
          terminals_list = terminals_list[:trajectories]
          timeouts_list = timeouts_list[:trajectories]
          weights_list = weights_list[:trajectories]
        num_trajectories = len(states_list)
        # Wrap for absorbing states
        if self.absorbing:
          absorbing_state, absorbing_action = torch.cat([torch.zeros(1, state_size), torch.ones(1, 1)], dim=1), torch.zeros(1, action_size)  # Create absorbing state and absorbing action
          for i in range(len(states_list)):
            # Append absorbing indicator (zero)
            states_list[i] = torch.cat([states_list[i], torch.zeros(states_list[i].size(0), 1)], dim=1)
            next_states_list[i] = torch.cat([next_states_list[i], torch.zeros(next_states_list[i].size(0), 1)], dim=1)
            if not timeouts_list[i][-1]:  # Apply for episodes that did not terminate due to time limits
              # Replace the final next state with the absorbing state and overwrite terminal status
              next_states_list[i][-1] = absorbing_state
              terminals_list[i][-1] = 0
              weights_list[i][-1] = 1 / subsample  # Importance weight absorbing state as kept during subsampling
              # Add absorbing state to absorbing state transition
              states_list[i] = torch.cat([states_list[i], absorbing_state], dim=0)
              actions_list[i] = torch.cat([actions_list[i], absorbing_action], dim=0)
              next_states_list[i] = torch.cat([next_states_list[i], absorbing_state], dim=0)
              terminals_list[i] = torch.cat([terminals_list[i], torch.zeros(1)], dim=0)
              timeouts_list[i] = torch.cat([timeouts_list[i], torch.zeros(1)], dim=0)
              weights_list[i] = torch.cat([weights_list[i], torch.full((1, ), 1 / subsample)], dim=0)  # Importance weight absorbing state as kept during subsampling
        # Subsample within trajectories
        if subsample > 1:
          for i in range(len(states_list)):
            # Subsample from random index in 0 to N-1 (procedure from original GAIL implementation)
            rand_start_idx, T = np.random.choice(subsample), len(states_list[i])
            idxs = range(rand_start_idx, T, subsample)
            if self.absorbing:
                # Subsample but keep absorbing state transitions
                idxs = sorted(list(set(idxs) | set([T - 2, T - 1])))
            states_list[i] = states_list[i][idxs]
            actions_list[i] = actions_list[i][idxs]
            next_states_list[i] = next_states_list[i][idxs]
            terminals_list[i] = terminals_list[i][idxs]
            timeouts_list[i] = timeouts_list[i][idxs]
            weights_list[i] = weights_list[i][idxs]

        transitions = {'states': torch.cat(states_list, dim=0), 'actions': torch.cat(actions_list, dim=0), 'next_states': torch.cat(next_states_list, dim=0), 'terminals': torch.cat(terminals_list, dim=0), 'timeouts': torch.cat(timeouts_list, dim=0), 'weights': torch.cat(weights_list, dim=0), 'num_trajectories': num_trajectories}
        # Pass 0 rewards to replay memory for interoperability/make sure reward information is not leaked to IL algorithm when data comes from an offline RL dataset
        transitions['rewards'] = torch.zeros_like(transitions['terminals'])

        return ReplayMemory(transitions['states'].size(0), state_size + (1 if self.absorbing else 0), action_size, self.absorbing, transitions=transitions)

class Get_modelstate(Node):

    def __init__(self):
        super().__init__('get_modelstate')
        self.subscription = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, data):
        global robot_pose

        unit_sphere_id = data.name.index('robot_1')
        robot_pose[0] = data.pose[unit_sphere_id].position.x
        robot_pose[1] = data.pose[unit_sphere_id].position.y

class Lidar_subscriber(Node):

    def __init__(self):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/robot_1_front/scan',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, data):
        global lidar_data
        # https://docs.ros.org/en/api/sensor_msgs/html/msg/LaserScan.html
        #gz_env.get_logger().info(f"len : {len(data.ranges)}")
        for i in range(20):
            lidar_data[i] = data.ranges[18*i]
            if(lidar_data[i] > 7.465):
                lidar_data[i] = 7.465
        #gz_env.get_logger().info(f"lidar_data : {lidar_data}")

# Evaluate agent with deterministic policy π
def evaluate_agent(actor: SoftActor, num_episodes: int, return_trajectories: bool=False) -> Union[Tuple[List[List[float]], Dict[str, Tensor]], List[List[float]]]:
  returns = []
  trajectories = []
  #if render:
  #    env.render()
  max_episode_steps = 300

  with torch.inference_mode():
    for _ in range(num_episodes):
      states = []
      actions = []
      rewards = []
      state = gz_env.reset()
      terminal = False
      t = 0
      while not terminal:
          action = actor.get_greedy_action(state)  # Take greedy action
          next_state, reward, terminal = gz_env.step(action, t, max_episode_steps)
          t += 1

          if return_trajectories:
            states.append(state)
            actions.append(action)
          rewards.append(reward)
          state = next_state
      returns.append(sum(rewards))

      if return_trajectories:
        # Collect trajectory data (including terminal signal, which may be needed for offline learning)
        terminals = torch.cat([torch.zeros(len(rewards) - 1), torch.ones(1)])
        trajectories.append({'states': torch.cat(states), 'actions': torch.cat(actions), 'rewards': torch.tensor(rewards, dtype=torch.float32), 'terminals': terminals})

  return (returns, trajectories) if return_trajectories else returns

if __name__ == '__main__':
    rclpy.init(args=None)
    
    with open(os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/conf/train_config.yaml', 'r') as yml:
        cfg = yaml.safe_load(yml)

    gz_env = GazeboEnv(cfg['imitation']['absorbing'], load_data=True)
    #eval_env = GazeboEnv(cfg['imitation']['absorbing'])
    get_modelstate = Get_modelstate()
    lidar_subscriber = Lidar_subscriber()

    action_list = []
    next_state_list = []
    state_list = []
    reward_list = []
    done_list = []
    time_out_list = []   

    assert cfg['defaults'][1]['algorithm'] in ['AdRIL', 'BC', 'DRIL', 'GAIL', 'GMMIL', 'PWIL', 'RED', 'SAC']
    cfg['memory']['size'] = min(cfg['steps'], cfg['memory']['size']) 
    assert cfg['bc_pretraining']['iterations'] >= 0
    assert cfg['imitation']['trajectories'] >= 0
    assert cfg['imitation']['mix_expert_data'] in ['none', 'mixed_batch', 'prefill_memory']
    if cfg['defaults'][1]['algorithm'] == 'AdRIL': 
      assert cfg['imitation']['mix_expert_data'] == 'mixed_batch'
    elif cfg['defaults'][1]['algorithm'] == 'DRIL': 
      assert 0 <= cfg.imitation.quantile_cutoff <= 1
    elif cfg['defaults'][1]['algorithm'] == 'GAIL':
      # Technically possible, but makes the control flow for training the discriminator more complicated
      assert cfg['imitation']['mix_expert_data'] != 'prefill_memory'
    '''
      assert cfg.imitation.discriminator.reward_function in ['AIRL', 'FAIRL', 'GAIL']
      assert cfg.imitation.grad_penalty >= 0
      assert cfg.imitation.entropy_bonus >= 0
      assert cfg.imitation.loss_function in ['BCE', 'Mixup', 'PUGAIL']
      if cfg.imitation.loss_function == 'Mixup': assert cfg.imitation.mixup_alpha > 0
      if cfg.imitation.loss_function == 'PUGAIL': assert 0 <= cfg.imitation.pos_class_prior <= 1 and cfg.imitation.nonnegative_margin >= 0
    '''
    assert cfg['logging']['interval'] >= 0
    normalization_max = 10
    normalization_min = -1

    # Load expert trajectories dataset
    expert_memory = gz_env.get_dataset(trajectories=cfg['imitation']['trajectories'], subsample=cfg['imitation']['subsample'])
    state_size = 23
    action_size = 3
    max_episode_steps = 300
    file_prefix = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/logs/'

    # Set up agent
    actor = SoftActor(state_size, action_size, cfg['reinforcement']['actor'])
    critic = TwinCritic(state_size, action_size, cfg['reinforcement']['critic'])
    log_alpha = torch.zeros(1, requires_grad=True)
    target_critic = create_target_network(critic)
    entropy_target = cfg['reinforcement']['target_temperature'] * action_size  # Entropy target heuristic from SAC paper for continuous action domains
    actor_optimiser = optim.AdamW(actor.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    critic_optimiser = optim.AdamW(critic.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    temperature_optimiser = optim.Adam([log_alpha], lr=cfg['training']['learning_rate'])
    memory = ReplayMemory(cfg['memory']['size'], state_size, action_size, cfg['imitation']['absorbing'])

    # Set up imitation learning components
    gz_env.get_logger().info(f"algorithm : {cfg['defaults'][1]['algorithm']}")
    if cfg['defaults'][1]['algorithm'] in ['AdRIL', 'DRIL', 'GAIL', 'GMMIL', 'PWIL', 'RED']:
      if cfg['defaults'][1]['algorithm'] == 'AdRIL':
        discriminator = RewardRelabeller(cfg['imitation']['update_freq'], cfg['imitation']['balanced'])  # Balanced sampling (switching between expert and policy data every update) is stateful
      if cfg['defaults'][1]['algorithm'] == 'DRIL':
        discriminator = SoftActor(state_size, action_size, cfg['imitation']['discriminator'])
      elif cfg['defaults'][1]['algorithm'] == 'GAIL':
        discriminator = GAILDiscriminator(state_size, action_size, cfg['imitation'], cfg['reinforcement']['discount'])
      elif cfg['defaults'][1]['algorithm'] == 'GMMIL':
        discriminator = GMMILDiscriminator(state_size, action_size, cfg['imitation'])
      elif cfg['defaults'][1]['algorithm'] == 'PWIL':
        discriminator = PWILDiscriminator(state_size, action_size, cfg['imitation'], expert_memory, max_episode_steps)
      elif cfg['defaults'][1]['algorithm'] == 'RED':
        discriminator = REDDiscriminator(state_size, action_size, cfg['imitation'])
      if cfg['defaults'][1]['algorithm'] in ['DRIL', 'GAIL', 'RED']:
        discriminator_optimiser = optim.AdamW(discriminator.parameters(), lr=cfg['imitation']['learning_rate'], weight_decay=cfg['imitation']['weight_decay'])

    # Metrics
    metrics = dict(train_steps=[], train_returns=[], test_steps=[], test_returns=[], test_returns_normalized=[], update_steps=[], predicted_rewards=[], alphas=[], entropies=[], Q_values=[])
    score = []  # Score used for hyperparameter optimization 

    if cfg['check_time_usage']: start_time = time.time()  # Performance tracking

    # Behavioural cloning pretraining
    if cfg['bc_pretraining']['iterations'] > 0:
      expert_dataloader = iter(cycle(DataLoader(expert_memory, batch_size=cfg['training']['batch_size'], shuffle=True, drop_last=True)))
      actor_pretrain_optimiser = optim.AdamW(actor.parameters(), lr=cfg['bc_pretraining']['learning_rate'], weight_decay=cfg['bc_pretraining']['weight_decay'])  # Create separate pretraining optimiser
      for _ in tqdm(range(cfg['bc_pretraining']['iterations']), leave=False):
        expert_transitions = next(expert_dataloader)
        behavioural_cloning_update(actor, expert_transitions, actor_pretrain_optimiser)

      if cfg['defaults'][1]['algorithm'] == 'BC':  # Return early if algorithm is BC
        if cfg['check_time_usage']:
            metrics['pre_training_time'] = time.time() - start_time
        test_returns = evaluate_agent(actor, cfg['evaluation']['episodes'])
        test_returns_normalized = (np.array(test_returns) - normalization_min) / (normalization_max - normalization_min)
        steps = [*range(0, cfg['steps'], cfg['evaluation']['interval'])]
        metrics['test_steps'] = [0]
        metrics['test_returns'] = [test_returns]
        metrics['test_returns_normalized'] = [list(test_returns_normalized)]
        lineplot(steps, len(steps) * [test_returns], filename=f'{file_prefix}test_returns', title=f"{cfg['defaults'][1]['algorithm']}: {cfg['env']}")

        torch.save(dict(actor=actor.state_dict()), f'{file_prefix}agent.pth')
        torch.save(metrics, f'{file_prefix}metrics.pth')
        #env.close()
        #eval_env.close()
        #return np.mean(test_returns_normalized)

    # Pretraining "discriminators"
    if cfg['defaults'][1]['algorithm'] in ['DRIL', 'RED']:
      expert_dataloader = iter(cycle(DataLoader(expert_memory, batch_size=cfg['training']['batch_size'], shuffle=True, drop_last=True)))
      for _ in tqdm(range(cfg.imitation.pretraining.iterations), leave=False):
        expert_transition = next(expert_dataloader)
        if cfg['defaults'][1]['algorithm'] == 'DRIL':
          behavioural_cloning_update(discriminator, expert_transition, discriminator_optimiser)  # Perform behavioural cloning updates offline on policy ensemble (dropout version)
        elif cfg['defaults'][1]['algorithm'] == 'RED':
          target_estimation_update(discriminator, expert_transition, discriminator_optimiser)  # Train predictor network to match random target network

      with torch.inference_mode():
        if cfg['defaults'][1]['algorithm'] == 'DRIL':
          discriminator.set_uncertainty_threshold(expert_memory['states'], expert_memory['actions'], cfg['imitation']['quantile_cutoff'])
        elif cfg['defaults'][1]['algorithm']== 'RED':
          discriminator.set_sigma(expert_memory['states'][:cfg['training']['batch_size']], expert_memory['actions'][:cfg['training']['batch_size']])  # Estimate on a minibatch for computational feasibility

      if cfg['check_time_usage']:
        metrics['pre_training_time'] = time.time() - start_time
        start_time = time.time()

      if cfg['imitation']['mix_expert_data'] == 'prefill_memory': memory.transfer_transitions(expert_memory)  # Once pretraining is over, transfer expert transitions to agent replay memory
    elif cfg['defaults'][1]['algorithm'] == 'PWIL':
      if cfg['imitation']['mix_expert_data'] != 'none':
        with torch.inference_mode():
          for i, transition in tqdm(enumerate(expert_memory), leave=False):
            expert_memory.rewards[i] = discriminator.compute_reward(transition['states'].unsqueeze(dim=0), transition['actions'].unsqueeze(dim=0))  # Greedily calculate the reward for PWIL for expert data and rewrite memory
            if transition['terminals'] or transition['timeouts']: discriminator.reset()  # Reset the expert data for PWIL
      if cfg['imitation']['mix_expert_data'] == 'prefill_memory': memory.transfer_transitions(expert_memory)  # Once rewards have been calculated, transfer expert transitions to agent replay memory
    elif cfg['defaults'][1]['algorithm'] == 'GMMIL':
      if cfg['imitation']['mix_expert_data'] == 'prefill_memory': memory.transfer_transitions(expert_memory)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(gz_env)
    executor.add_node(get_modelstate)
    executor.add_node(lidar_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = gz_env.create_rate(2)

    try:
        while rclpy.ok():

            # Training
            t = 0
            state = gz_env.reset()
            terminal = False
            train_return = 0
            if cfg['defaults'][1]['algorithm'] in ['GAIL', 'RED']:
                discriminator.eval()  # Set the "discriminator" to evaluation mode (except for DRIL, which explicitly uses dropout)

            pbar = tqdm(range(1, cfg['steps'] + 1), unit_scale=1, smoothing=0)
            for step in pbar:
                # Collect set of transitions by running policy π in the environment
                with torch.inference_mode():
                    action = actor(state).sample()
                    next_state, reward, terminal = gz_env.step(action, t, max_episode_steps)
                    t += 1
                    train_return += reward
                    if cfg['defaults'][1]['algorithm'] == 'PWIL':
                        # Greedily calculate the reward for PWIL
                        reward = discriminator.compute_reward(state, action)
                    memory.append(step, state, action, reward, next_state, terminal and t != max_episode_steps, t == max_episode_steps)  # True reward stored for SAC, should be overwritten by IL algorithms; if env terminated due to a time limit then do not count as terminal (store as timeout)
                    state = next_state

                # Reset environment and track metrics on episode termination
                if terminal:  # If terminal (or timed out)
                    gz_env.get_logger().info("terminal")
                    if cfg['imitation']['absorbing'] and t != max_episode_steps:
                        memory.wrap_for_absorbing_states()  # Wrap for absorbing state if terminated without time limit
                    if cfg['defaults'][1]['algorithm'] == 'PWIL':
                        discriminator.reset()  # Reset the expert data for PWIL
                    # Store metrics and reset environment
                    metrics['train_steps'].append(step)
                    metrics['train_returns'].append([train_return])
                    pbar.set_description(f'Step: {step} | Return: {train_return}')
                    t = 0
                    state = gz_env.reset()
                    train_return = 0

                # Train agent and imitation learning component
                if step >= cfg['training']['start'] and step % cfg['training']['interval'] == 0:
                  # Sample a batch of transitions
                  transitions, expert_transitions = memory.sample(cfg['training']['batch_size']), expert_memory.sample(cfg['training']['batch_size'])

                  if cfg['defaults'][1]['algorithm'] in ['AdRIL', 'DRIL', 'GAIL', 'GMMIL', 'RED']:  # Note that PWIL predicts and stores rewards online during environment interaction
                    # Train discriminator
                    if cfg['defaults'][1]['algorithm'] == 'GAIL':
                      discriminator.train()
                      adversarial_imitation_update(actor, discriminator, transitions, expert_transitions, discriminator_optimiser, cfg['imitation'])
                      discriminator.eval()

                    # Optionally, mix expert data into agent data for training
                    if cfg['imitation']['mix_expert_data'] == 'mixed_batch' and cfg['defaults'][1]['algorithm'] != 'AdRIL':
                        mix_expert_agent_transitions(transitions, expert_transitions)
                    # Predict rewards
                    states = transitions['states']
                    actions = transitions['actions']
                    next_states = transitions['next_states']
                    terminals = transitions['terminals']
                    weights = transitions['weights']
                    
                    # Note that using the entire dataset is prohibitively slow in off-policy case (for relevant algorithms)
                    expert_states = expert_transitions['states']
                    expert_actions = expert_transitions['actions']
                    expert_next_states = expert_transitions['next_states']
                    expert_terminals = expert_transitions['terminals']
                    expert_weights = expert_transitions['weights']

                    with torch.inference_mode():
                      if cfg['defaults'][1]['algorithm'] == 'AdRIL':
                        # Uses a mix of expert and policy data and overwrites transitions (including rewards) inplace
                        discriminator.resample_and_relabel(transitions, expert_transitions, step, memory.num_trajectories, expert_memory.num_trajectories)  
                      elif cfg['defaults'][1]['algorithm'] == 'DRIL':
                        transitions['rewards'] = discriminator.predict_reward(states, actions)
                      elif cfg['defaults'][1]['algorithm'] == 'GAIL':
                        transitions['rewards'] = discriminator.predict_reward(**make_gail_input(states, actions, next_states, terminals, actor,
                                                                                                cfg['imitation']['discriminator']['reward_shaping'],
                                                                                                cfg['imitation']['discriminator']['subtract_log_policy']))
                      elif cfg['defaults'][1]['algorithm'] == 'GMMIL':
                        transitions['rewards'] = discriminator.predict_reward(states, actions, expert_states, expert_actions, weights, expert_weights)
                      elif cfg['defaults'][1]['algorithm'] == 'RED':
                        transitions['rewards'] = discriminator.predict_reward(states, actions)

                  # Perform a behavioural cloning update (optional)
                  if cfg['imitation']['bc_aux_loss']:
                      behavioural_cloning_update(actor, expert_transitions, actor_optimiser)
                  # Perform a SAC update
                  log_probs, Q_values = sac_update(actor, critic, log_alpha, target_critic, transitions,
                                                   actor_optimiser, critic_optimiser, temperature_optimiser,
                                                   cfg['reinforcement']['discount'], entropy_target, cfg['reinforcement']['polyak_factor'])
                  # Save auxiliary metrics
                  if cfg['logging']['interval'] > 0 and step % cfg['logging']['interval'] == 0:
                    gz_env.get_logger().info("Saving auxiliary metrics")
                    metrics['update_steps'].append(step)
                    metrics['predicted_rewards'].append(transitions['rewards'].numpy())
                    metrics['alphas'].append(log_alpha.exp().detach().numpy())
                    metrics['entropies'].append((-log_probs).numpy())  # Actions are sampled from the policy distribution, so "p" is already included
                    metrics['Q_values'].append(Q_values.numpy())

                # Evaluate agent and plot metrics
                if step % cfg['evaluation']['interval'] == 0 and not cfg['check_time_usage']:
                  gz_env.get_logger().info("Evaluation of the agent")
                  test_returns = evaluate_agent(actor, cfg['evaluation']['episodes'])
                  test_returns_normalized = (np.array(test_returns) - normalization_min) / (normalization_max - normalization_min)
                  score.append(np.mean(test_returns_normalized))
                  metrics['test_steps'].append(step)
                  metrics['test_returns'].append(test_returns)
                  metrics['test_returns_normalized'].append(list(test_returns_normalized))
                  '''
                  lineplot(metrics['test_steps'], metrics['test_returns'], filename=f"{file_prefix}test_returns", title=f"{cfg['defaults'][1]['algorithm']}: {cfg['env']} Test Returns")
                  if len(metrics['train_returns']) > 0:  # Plot train returns if any
                    lineplot(metrics['train_steps'], metrics['train_returns'], filename=f"{file_prefix}train_returns", title=f"Training {cfg['defaults'][1]['algorithm']}: {cfg['env']} Train Returns")
                  if cfg['logging']['interval'] > 0 and len(metrics['update_steps']) > 0:
                    if cfg['defaults'][1]['algorithm'] != 'SAC':
                        lineplot(metrics['update_steps'], metrics['predicted_rewards'], filename=f'{file_prefix}predicted_rewards', yaxis='Predicted Reward', title=f"{cfg['defaults'][1]['algorithm']}: {cfg['env']} Predicted Rewards")
                    lineplot(metrics['update_steps'], metrics['alphas'], filename=f'{file_prefix}sac_alpha', yaxis='Alpha', title=f"{cfg['defaults'][1]['algorithm']}: {cfg['env']} Alpha")
                    lineplot(metrics['update_steps'], metrics['entropies'], filename=f'{file_prefix}sac_entropy', yaxis='Entropy', title=f"{cfg['defaults'][1]['algorithm']}: {cfg['env']} Entropy")
                    lineplot(metrics['update_steps'], metrics['Q_values'], filename=f'{file_prefix}Q_values', yaxis='Q-value', title=f"{cfg['defaults'][1]['algorithm']}: {cfg['env']} Q-values")
                  '''

            gz_env.get_logger().info(f"metrics:{metrics}")

            if cfg['check_time_usage']:
                metrics['training_time'] = time.time() - start_time

            if cfg['save_trajectories']:
                # Store trajectories from agent after training
                _, trajectories = evaluate_agent(actor, cfg['evaluation']['episodes'], return_trajectories=True, render=cfg['render'])
                torch.save(trajectories, f'{file_prefix}trajectories.pth')
            # Save agent and metrics
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), log_alpha=log_alpha), f'{file_prefix}agent.pth')
            if cfg['defaults'][1]['algorithm'] in ['DRIL', 'GAIL', 'RED']:
                torch.save(discriminator.state_dict(), f'{file_prefix}discriminator.pth')
            torch.save(metrics, f'{file_prefix}metrics.pth')

            score = np.mean(score)
            break

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    rclpy.shutdown()
    executor_thread.join()
