# 奖励函数：局部奖励、全局奖励

import numpy as np


# def calculate_local_reward(camera_id, coverage_scores, visibility_matrix, lambda_weight=0.2):
#     '''
#     计算相机i的局部奖励:
#     局部奖励 = 被感知的覆盖目标得分之和 + 方向对齐奖励

#     方向对齐奖励还没设计好，暂时没有加入

#     camera_id:相机id
#     cameras:相机位置列表
#     targets:目标位置列表
#     coverage_scores:覆盖评分列表
#     visibility_matrixs:可视矩阵
#     lambda_weight:方向对齐奖励权重
#     '''
#     reward = 0
#     for j, r_j in enumerate(coverage_scores):
#         if visibility_matrix[camera_id][j] != 1:
#             continue
#         w_j = 1 - 1 / (1 + np.exp(-r_j)) # 系数加权项
#         reward += w_j * r_j
#     return reward

# def calculate_global_reward(coverage_scores, visibility_matrix, lambda_weight=0.5):
#     '''
#     计算系统全局奖励
#     全局奖励 = 平均目标覆盖得分 + 全局覆盖率

#     coverage_scores: 覆盖评分列表
#     visibility_matrix: 可见性矩阵
#     lambda_weight: 权重参数

#     '''
#     avg_score = np.mean(coverage_scores)

#     covered = np.any(visibility_matrix, axis=0) # 每个目标是否至少被一个相机看到
#     covergae_rate = np.sum(covered) / len(coverage_scores)

#     global_reward = lambda_weight * avg_score + (1 - lambda_weight) * covergae_rate
#     return global_reward


class RewardFunction:
    def __init__(self, mode='potential', lambda_cover=1.0, lambda_align=0.3, transition_steps=50000):
        self.mode = mode # mode:奖励函数模式 'potential'(第一阶段), 'hybrid'(第二阶段), 'soft-transition'(软过渡)
        self.step = 0 # 步数
        if mode == 'hybrid':
            self.lambda_cover = lambda_cover
            self.lambda_align = lambda_align
        elif mode == 'soft-transition':
            self.lambda_cover = lambda_cover
            self.lambda_align = lambda_align
            self.transition_steps = transition_steps
        
    def step_update(self):
        '''步数更新'''
        self.step += 1
    
    def calculate_potential_reward(self, camera_id, camera_potential):
        '''第一阶段奖励：势能'''
        return camera_potential[camera_id]
    
    def calculate_hybrid_reward(self, camera_id, visibility_matrix):
        covered = np.any(visibility_matrix, axis=0)
        coverage_rate = np.sum(covered) / len(visibility_matrix[0]) # 全局覆盖率

        align_reward = 0 # 方向奖励暂时没写
        
        return self.lambda_cover * coverage_rate + self.lambda_align * align_reward
    
    def calculate_alpha(self):
        '''软过渡获取alpha值'''
        return min(1.0, self.step / self.transition_steps)
    
    def calculate_transition_reward(self, camera_id, camera_potential, visibility_matrix):
        '''软过渡奖励'''
        alpha = self.calculate_alpha()
        potential_reward = self.calculate_potential_reward(camera_id, camera_potential)
        hybrid_reward = self.calculate_hybrid_reward(camera_id, visibility_matrix)
        reward = (1 - alpha) * potential_reward + alpha * hybrid_reward

        self.step_update()
        
        return reward
    
class Reward_Cover_Only:
    def __init__(self, lambda_global=0.1):
        self.lambda_global = lambda_global
    
    def calculate_reward(self, visibility_matrix):
        covered = np.any(visibility_matrix, axis=0)  # 每个目标是否至少被一个相机看到
        coverage_rate = np.sum(covered) / len(visibility_matrix[0])
        if coverage_rate == 0:
            local_reward = np.zeros(visibility_matrix.shape[0])
            global_reward = -1
        else:
            global_reward = coverage_rate
            local_reward = np.sum(visibility_matrix, axis=1) / (np.sum(covered))
        reward = self.lambda_global * global_reward + (1 - self.lambda_global) * local_reward
        return reward
    

class Reward_QualityDeltaGlobal:
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, alpha_dist=1.0):
        """
        alpha: 感知质量奖励权重
        beta: 首次覆盖奖励权重
        gamma: 全局覆盖率奖励权重
        alpha_dist: 距离惩罚的指数系数
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.alpha_dist = alpha_dist

    def calculate_reward(self, visibility_matrix, global_state, global_covered_mask_prev):
        """
        visibility_matrix: np.array [N, G]，相机是否能看到目标（0/1）
        global_state: np.array [N, G, 2]，每个目标的 (距离d, 相对角度theta)，theta单位为弧度
        global_covered_mask_prev: np.array [G]，上一帧中每个目标是否已被覆盖（0/1）

        返回：
        - reward: np.array [N]，每个相机的最终奖励
        - global_covered_mask: np.array [G]，当前帧中目标的覆盖掩码（供下一帧用）
        """
        N, M = visibility_matrix.shape  # N: 相机数量, M: 目标数量
        reward = np.zeros(N) # 每个相机的奖励
        quality_reward = np.zeros(N) # 感知质量奖励
        delta_reward = np.zeros(N) # 首次覆盖奖励

        global_covered_mask = np.any(visibility_matrix, axis=0)  # 当前帧中每个目标是否被覆盖

        # ---- 1. 感知质量奖励 ----
        for i in range(N):
            for j in range(M):
                if visibility_matrix[i, j] == 1:
                    d, theta = global_state[i, j]
                    quality = np.exp(-self.alpha_dist * d) * np.cos(theta) # 感知质量
                    quality_reward[i] += quality
        
        # ---- 2. 首次覆盖奖励 ----
        delta_covered = np.logical_and(global_covered_mask, np.logical_not(global_covered_mask_prev)) # [M]
        for i in range(N):
            for j in range(M):
                if visibility_matrix[i, j] == 1 and delta_covered[j]:
                    delta_reward[i] += 1.0
        
        # ---- 3. 全局覆盖率奖励 每个agent都一样----
        global_reward = np.sum(global_covered_mask) / M if M > 0 else 0.0

        # ---- 4. 最终奖励计算 ----
        reward = (self.alpha * quality_reward + 
                  self.beta * delta_reward + 
                  self.gamma * global_reward)
        
        return reward, global_covered_mask 
    
class Reward_Ultra_Simple:
    def __init__(self):
        pass
    
    def calculate_reward(self, visibility_matrix, global_state, global_covered_mask_prev):
        """超简化：只返回覆盖率"""
        coverage_rate = np.mean(np.any(visibility_matrix, axis=0))
        reward = np.full(visibility_matrix.shape[0], coverage_rate)
        global_covered_mask = np.any(visibility_matrix, axis=0)
        return reward, global_covered_mask