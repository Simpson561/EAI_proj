import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F

class EnergyPPO(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Energy network - E(s,a)
        self.energy_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Value network - V(s)
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Temperature parameter (log alpha to ensure positivity)
        self.log_temperature = nn.Parameter(torch.zeros(1))
        
        # Action bounds for scaling
        self.action_scale = torch.tensor(1.0)
        self.action_bias = torch.tensor(0.0)
        
    def set_action_bounds(self, low, high):
        """设置动作空间的范围"""
        self.action_scale = (high - low) / 2.0
        self.action_bias = (high + low) / 2.0
        
    def scale_action(self, action):
        """将[-1,1]范围的动作缩放到实际范围"""
        return action * self.action_scale + self.action_bias
        
    def unscale_action(self, action):
        """将实际范围的动作缩放回[-1,1]"""
        return (action - self.action_bias) / self.action_scale

    def compute_energy(self, states, actions):
        """计算能量函数 E(s,a)"""
        x = torch.cat([states, actions], dim=-1)
        return self.energy_net(x)

    def get_action_distribution(self, states, num_samples=50):
        """通过能量采样获得动作分布"""
        batch_size = states.shape[0]
        
        # 生成随机动作样本
        random_actions = torch.randn(num_samples, batch_size, self.action_dim, 
                                   device=states.device)
        expanded_states = states.unsqueeze(0).expand(num_samples, -1, -1)
        
        # 计算每个样本的能量
        energies = self.compute_energy(expanded_states, random_actions)
        
        # 使用softmax计算概率权重
        temperature = torch.exp(self.log_temperature)
        weights = F.softmax(-energies / temperature, dim=0)
        
        # 计算加权平均和方差
        mean = (weights * random_actions).sum(dim=0)
        var = (weights * (random_actions - mean.unsqueeze(0))**2).sum(dim=0)
        std = torch.sqrt(var + 1e-6)
        
        return Normal(mean, std)

    def get_action(self, states, deterministic=False):
        """采样或确定性选择动作"""
        distribution = self.get_action_distribution(states)
        
        if deterministic:
            actions = distribution.mean
        else:
            actions = distribution.rsample()
            
        # 将动作限制在[-1,1]范围内
        actions = torch.tanh(actions)
        # 缩放到实际动作范围
        scaled_actions = self.scale_action(actions)
        
        return scaled_actions

    def evaluate_actions(self, states, actions):
        """评估给定的状态-动作对"""
        unscaled_actions = self.unscale_action(actions)
        distribution = self.get_action_distribution(states)
        
        # 计算对数概率
        log_probs = distribution.log_prob(unscaled_actions).sum(-1)
        
        # 计算熵
        entropy = distribution.entropy().mean()
        
        # 计算状态值
        value = self.value_net(states).squeeze()
        
        return log_probs, entropy, value

    def get_value(self, states):
        """获取状态值估计"""
        return self.value_net(states).squeeze()

class PPOTrainer:
    def __init__(self, 
                 agent,
                 optimizer,
                 clip_ratio=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 target_kl=0.015):
        
        self.agent = agent
        self.optimizer = optimizer
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef 
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

    def train_step(self, states, actions, old_log_probs, advantages, returns):
        """执行一次PPO更新"""
        # 评估当前动作
        new_log_probs, entropy, values = self.agent.evaluate_actions(states, actions)
        
        # 计算概率比
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 计算PPO裁剪目标
        obj1 = ratio * advantages
        obj2 = ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(obj1, obj2).mean()
        
        # 值函数损失
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        
        # 温度参数损失(用于自动调节探索)
        temp_loss = -self.agent.log_temperature.exp() * entropy
        
        # 总损失
        loss = (policy_loss + 
                self.value_coef * value_loss + 
                self.entropy_coef * temp_loss)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'temperature': self.agent.log_temperature.exp().item()
        }