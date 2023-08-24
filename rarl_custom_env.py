### RARL with custom environment
# Author: Uday Kiran umanduri@buffalo.edu udaykiran2495@gmail.com
# Date: 10 Aug 2023
# Original Paper: Pinto, Lerrel, et al. "Robust adversarial reinforcement learning."
#                 International Conference on Machine Learning. PMLR, 2017. https://arxiv.org/pdf/1703.02702v1.pdf 
# original implementation: https://github.com/lerrel/gym-adv
# code reference: https://github.com/Jekyll1021/RARL/blob/master/reimplementation/PPO_RARL.py

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mujoco_py
from torch.distributions.normal import Normal
import random
import time
import wandb
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces
import os

from mujoco_py import functions as mjcf

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

HALF_CHEETAH_XML = """
<?xml version="1.0" ?>
<mujoco model="cheetah">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
  <default>
    <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <size nstack="300000" nuser_geom="1"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 .7">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="-.5 0 0 .5 0 0" name="torso" size="0.046" type="capsule"/>
      <geom axisangle="0 1 0 .87" name="head" pos=".6 0 .1" size="0.046 .15" type="capsule"/>
      <!-- <site name='tip'  pos='.15 0 .11'/>-->
      <body name="bthigh" pos="-.5 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom axisangle="0 1 0 -3.8" name="bthigh" pos=".1 0 -.13" size="0.046 .145" type="capsule"/>
        <body name="bshin" pos=".16 0 -.25">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom axisangle="0 1 0 -2.03" name="bshin" pos="-.14 0 -.07" rgba="0.9 0.6 0.6 1" size="0.046 .15" type="capsule"/>
          <body name="bfoot" pos="-.28 0 -.14">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom axisangle="0 1 0 -.27" name="bfoot" pos=".03 0 -.097" rgba="0.9 0.6 0.6 1" size="0.046 .094" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="fthigh" pos=".5 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1 .7" stiffness="180" type="hinge"/>
        <geom axisangle="0 1 0 .52" name="fthigh" pos="-.07 0 -.12" size="0.046 .133" type="capsule"/>
        <body name="fshin" pos="-.14 0 -.24">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 .87" stiffness="120" type="hinge"/>
          <geom axisangle="0 1 0 -.6" name="fshin" pos=".065 0 -.09" rgba="0.9 0.6 0.6 1" size="0.046 .106" type="capsule"/>
          <body name="ffoot" pos=".13 0 -.18">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-.5 .5" stiffness="60" type="hinge"/>
            <geom axisangle="0 1 0 -.6" name="ffoot" pos=".045 0 -.07" rgba="0.9 0.6 0.6 1" size="0.046 .07" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor gear="120" joint="bthigh" name="bthigh"/>
    <motor gear="90" joint="bshin" name="bshin"/>
    <motor gear="60" joint="bfoot" name="bfoot"/>
    <motor gear="120" joint="fthigh" name="fthigh"/>
    <motor gear="60" joint="fshin" name="fshin"/>
    <motor gear="30" joint="ffoot" name="ffoot"/>
  </actuator>
</mujoco>
"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class HalfCheetahMod(gym.Env):
    
    def __init__(self, name='HalfCheetahMod-v0'):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
       
        self.dt = 0.01
        
        self.mujoco_model = mujoco_py.load_model_from_xml(HALF_CHEETAH_XML)
        self.sim = mujoco_py.MjSim(self.mujoco_model) 
        
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        
        self._adv_body_index = self.get_adv_bindex() # [1, 4, 7]
        
        self.policy_protg = Policy(env=self)
        self.policy_adv = Policy(env=self)
        
    def get_adv_bindex(self):
        _adv_f_bname = ['torso', 'bfoot', 'ffoot']
        bnames = self.mujoco_model.body_names
        _adv_body_index = [0, 0, 0]
        _adv_body_index[0] = bnames.index(_adv_f_bname[0])
        _adv_body_index[1] = bnames.index(_adv_f_bname[1])
        _adv_body_index[2] = bnames.index(_adv_f_bname[2])
        return _adv_body_index
        
    def apply_adv(self, action):
        # new_xfrc = self.sim.data.xfrc_applied*0.0
        # new_xfrc[self._adv_body_index] = np.array([action[0], 0., action[1], 0., 0., 0.])
        
        # sim_state = self.sim.get_state()
        # print("========= sim_state ", sim_state)
        # sim_state.xfrc_applied = new_xfrc
        # self.sim.set_state(sim_state)
        
        # self.sim.forward()
        self.sim.data.xfrc_applied[self._adv_body_index] = np.array([action[0], 0., action[1], 0., 0., 0.])
        return 
    
    def step(self, action_protg, action_adv):
        xposbefore = self.sim.data.qpos[0]
        self.apply_adv(action_adv)
        self.do_simulation(action_protg)
        xposafter = self.sim.data.qpos[0]
        observation = self._get_obs()
        reward_ctrl = -0.1 * np.square(action_protg).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        
        done = False
        if self.sim.data.qvel[0] < 0.2:
            done = True

        return observation, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)
    
    def do_simulation(self, action, n_frames=1024):
        self.sim.data.ctrl[:] = action
        for _ in range(n_frames):
            self.sim.step()
        
    def _get_obs(self):
        obs = np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )
        obs = torch.from_numpy(obs).float().to(device)
        return obs
    
    def reset(self):
        return self.reset_model()
    
    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(
            low=-0.1, high=0.1, size=self.sim.model.nq
        )
        qvel = self.init_qvel + np.random.standard_normal(self.sim.model.nv) * 0.1
        
        sim_state = self.sim.get_state()
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = qvel
        
        self.sim.set_state(sim_state)
        self.sim.forward()
        
        return self._get_obs()

gym.register(
    id='HalfCheetahMod-v0',
    entry_point=HalfCheetahMod,
)

class Policy(nn.Module):
    def __init__(self, env=None):
        super().__init__()
        self.observation_space_shape = env.observation_space.shape
        self.action_space_shape = env.action_space.shape
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(self.observation_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(self.action_space_shape)), std=0.01),
        )
        
    def forward(self, x):
        output = self.network(x)
        mean = output.mean().item()
        std = output.std().item()
        return output, mean, std
    
def wrap_env(env):
    
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, f"videos/{env.spec.id}")
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=0.99)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

def rollout(env, policy_protg, policy_adv, N_traj):
    _actions_protg = [ [] for _ in range(N_traj) ]
    _actions_adv = [ [] for _ in range(N_traj) ]
    _observations = [ [] for _ in range(N_traj) ]
    
    _rewards_protg = [ [] for _ in range(N_traj) ]
    _rewards_adv = [ [] for _ in range(N_traj) ]
    
    actions_adv = torch.zeros((N_traj, 1)  + env.action_space.shape).to(device)
    actions_protg = torch.zeros((N_traj, 1)  + env.action_space.shape).to(device)
    observations = torch.zeros((N_traj, 1)  + env.observation_space.shape).to(device)
    
    trajectories = []
    
    next_observation = env.reset().to(device)
    
    for i in range(1, N_traj+1):
        
        traj_reward_protg = 0
        
        while True:
            next_action_adv, _, _ = policy_adv(next_observation)
            # next_action_adv = next_action_adv.detach().cpu().numpy()
            
            next_action_protg, _, _ = policy_protg(next_observation)
            # next_action_protg = next_action_protg.detach().cpu().numpy()
            
            next_observation, reward, done, _ = env.step(next_action_protg, next_action_adv)
            traj_reward_protg += reward
            
            
            _observations[i-1].append(next_observation)
            
            next_action_adv = torch.from_numpy(next_action_adv).to(device)
            _actions_adv[i-1].append(next_action_adv)
            next_action_protg = torch.from_numpy(next_action_protg).to(device)
            _actions_protg[i-1].append(next_action_protg)
            
            _rewards_protg[i-1].append(reward)
            _rewards_adv[i-1].append(-reward)
            
            
            if done:
                next_observation = env.reset().to(device)
                
                break
    
    
    print("====== _observations ", len(_observations), type(_observations))
    print("====== _actions_protg ", len(_actions_protg), type(_actions_protg))
    print("====== _actions_adv ", len(_actions_adv), type(_actions_adv))
    print("====== _rewards_protg ", len(_rewards_protg), type(_rewards_protg))
    print("====== _rewards_adv ", len(_rewards_adv), type(_rewards_adv))
    
            
    return observations, actions_protg, actions_adv, rewards_protg, rewards_adv

def policyOptimizer(observations, policy, rewards, N_traj, learning_rate=1e-3):
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    print("====== policyOptimizer ======")
    print(observations.shape)
    print(rewards.shape)
    print(policy)
    
    output = torch.zeros((N_traj, 1)  + env.action_space.shape).to(device)
    for i in range(1, N_traj+1):
        output[i-1] = policy(observations[i-1])
    
    
    print("====== output ", output.shape)
    print("====== rewards ", rewards.shape)
    
    optimizer.zero_grad()
    
    loss = nn.MSELoss(output, rewards)
    loss.backward()
    optimizer.step()
    

        
    return policy

if __name__ == '__main__':
    seed = 42
    num_steps = 2048
    total_timesteps = 1e6
    batch_size = num_steps
    num_minibatches = 32
    minibatch_size = batch_size // num_minibatches
    update_epochs = 10
    num_updates = total_timesteps // batch_size
    
    N_iter = 5000
    N_protg = 32
    N_adv = 32
    N_traj = 100
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    env = gym.make('HalfCheetahMod-v0')
    
    writer = SummaryWriter(f"runs/{env.spec.id}")
    writer.add_text("env", str(env))
    # TODO vectorized envs
    assert isinstance(env.observation_space, gym.spaces.Box), "only continuous action space is supported"
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"
    start_time = time.time()
    
    policy_protg = env.policy_protg.to(device)
    policy_adv = env.policy_adv.to(device)
    
    for update in range(1, N_iter+1):
        
        for j in range(1, N_protg):
            observations, actions_protg, _, rewards_protg, _ = rollout(env, policy_protg, policy_adv, N_traj)
            policy_protg = policyOptimizer(observations, policy_protg, rewards_protg, N_traj)
            
        for j in range(1, N_adv):
            observations, _, actions_adv, _, rewards_adv = rollout(env, policy_protg, policy_adv, N_traj)
            policy_adv = policyOptimizer(observations, policy_adv, rewards_adv, N_traj)
            
        
            

            
    
            
            
            
            
            
            
            
            
            
    
    
    
    
    