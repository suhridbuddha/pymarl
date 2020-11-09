from .multiagentenv import MultiAgentEnv
from pettingzoo.sisl import pursuit_v1, waterworld_v1, multiwalker_v3
from pettingzoo.butterfly import knights_archers_zombies_v2, pistonball_v0
from supersuit import action_lambda_v0,flatten_v0,resize_v0
import supersuit
import json
import random
import os
import gym
import numpy as np

def make_env(env_name):
    if env_name == "pistonball":
        env = pistonball_v0.parallel_env(max_frames=100)
        env = supersuit.resize_v0(env, 16, 16)
        env = supersuit.dtype_v0(env,np.float32)
        env = supersuit.normalize_obs_v0(env)
        return supersuit.flatten_v0(env)
    if env_name == "KAZ":
        env = knights_archers_zombies_v2.parallel_env(max_frames=100)
        env = supersuit.resize_v0(env, 32, 32)
        env = supersuit.dtype_v0(env,np.float32)
        env = supersuit.normalize_obs_v0(env)
        return supersuit.flatten_v0(env)
    if env_name == "pursuit":
        env = pursuit_v1.parallel_env(max_frames=100)
        env = supersuit.resize_v0(env, 32, 32)
        return supersuit.flatten_v0(env)
    elif env_name == "waterworld":
        return waterworld_v1.parallel_env(max_frames=100)
    elif env_name == "multiwalker":
        return multiwalker_v3.parallel_env(max_frames=100)
    else:
        raise RuntimeError("bad environment name")

class PettingZooEnv(MultiAgentEnv):
    def __init__(self, env_name, seed=None):
        self.env_name = env_name
        self.env = make_env(env_name)#.env()
        act_space = self.env.action_spaces[self.env.possible_agents[0]]
        if isinstance(act_space, gym.spaces.Box):
            self.num_discrete_acts = 31
            self.all_actions = [act_space.sample() for _ in range(self.num_discrete_acts)]
        else:
            self.num_discrete_acts = act_space.n
            self.all_actions = list(range(self.num_discrete_acts))

        # self.env = action_lambda_v0(self.env,
        #     lambda action, space : random_acts[action],
        #     lambda space: gym.spaces.Discrete(num_discrete_acts))
        # self.env = flatten_v0(self.env)
        self.env.reset()
        self.actions = []
        self.episode_limit = 105#self.env.max_frames
        self.n_agents = self.env.num_agents

    def step(self, actions):
        action_dict = {agent: self.all_actions[act] for agent, act in zip(self.env.possible_agents, actions)}
        obs, rews, dones, infos = self.env.step(action_dict)#, observe=False)
        # print(dones)
        self.actions[-1].append(actions)
        self.ep_len += 1
        self.observations = [obs[agent] if agent in dones else np.zeros_like(self.env.observation_spaces[agent].low) for agent in self.env.possible_agents]
        return sum(rews.values())/self.env.num_agents, all(dones.values()) or self.ep_len >= 500, {}

    def get_stats(self):
        return {}

    def get_obs(self):
        return [self.get_obs_agent(i) for i in range(self.env.max_num_agents)]

    def get_obs_agent(self, agent_id):
        # print(self.observations[agent_id].flatten().shape)
        # print(self.get_obs_size())
        return self.observations[agent_id].flatten()

    def get_obs_size(self):
        return int(np.prod(next(iter(self.env.observation_spaces.values())).shape))

    def get_state(self):
        return np.concatenate([self.get_obs_agent(o) for o in range(self.env.max_num_agents)],axis=0)

    def get_state_size(self):
        return  self.get_obs_size()*self.env.num_agents

    def get_avail_actions(self):
        return [[1]*self.get_total_actions()]*self.n_agents

    def get_avail_agent_actions(self, agent_id):
        return [1]*self.get_total_actions()

    def get_total_actions(self):
        return self.num_discrete_acts

    def reset(self):
        obs = self.env.reset()
        self.ep_len = 0
        self.actions.append([])
        if len(self.actions) > 10:
            self.actions.pop(0)
        self.observations = [obs[agent] for agent in self.env.agents]
        return self.get_obs(), self.get_state()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self):
        pass#raise NotImplementedError

    def save_replay(self):
        data = json.dumps(self.actions)
        os.makedirs("results/save_replay",exist_ok=True)
        with open(f"results/{self.env_name}.json") as file:
            file.write(data)

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
