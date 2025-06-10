import math
import os
import gym
import json
from gym import spaces
import numpy as np
from .potential_field import calculate_attract_potential, calculate_repel_potential
from .reward_fuctions import RewardFunction, Reward_Cover_Only
from .render import render

class Pose_Env_Base(gym.Env):
    def __init__(self, args, config_name='env.json', setting_path=None):
        '''
        setting_path: 是否单独设置配置json文件的路径
        
        '''
        super(Pose_Env_Base, self).__init__()

        self.env_path = 'MSMTC/DigitalPose2D'
        if setting_path:
            self.setting_path = setting_path
        else:
            self.setting_path = os.path.join(self.env_path, config_name)

        with open(self.setting_path, encoding='utf-8') as f: # 读取配置json文件
            setting = json.load(f)

        '''参数获取'''
        self.num_episodes = 0 # episode数量计数器
        self.steps = 0 # 步数计数器
        if args.max_steps == -1: # 每个episode最大步数
            self.max_steps = setting['max_steps']
        else:
            self.max_steps = args.max_steps

        if args.obstacle_flag == -1: # 是否有障碍物
            self.obstacle_flag = setting['obstacle_flag']
        else:
            self.obstacle_flag = args.obstacle_flag

        if args.target_move_type == -1: # 目标移动类型
            self.target_move_type = setting['target_move_type']
        else:
            self.target_move_type = args.target_move_type

        if args.num_cameras == -1: # 摄像头数量
            self.num_cameras = setting['num_cameras']
        else:
            self.num_cameras = args.num_cameras
        
        if args.num_targets == -1: # 目标数量
            self.num_targets = setting['num_targets']
        else:
            self.num_targets = args.num_targets
        
        if args.visual_distance == -1: # 相机可视距离
            self.visual_distance = setting['visual_distance']
        else:
            self.visual_distance = args.visual_distance
        
        if args.visual_angle == -1: # 相机监控角度
            self.visual_angle = setting['visual_angle']
        else:
            self.visual_angle = args.visual_angle
        
        if args.rotation_scale == -1: # 每个动作的旋转角度
            self.rotation_scale = setting['rotation_scale']
        else:
            self.rotation_scale = args.rotation_scale

        if args.move_scale == -1: # 每个动作的位移距离
            self.move_scale = setting['move_scale']
        else:
            self.move_scale = args.move_scale

        # if args.max_visual_num == -1: # 相机最大可视目标数量（因为做了变长的）
        #     self.max_visual_num = setting['max_visual_num'] 
        # else:
        #     self.max_visual_num = args.max_visual_num 

        self.area = setting['area'] # 区域（Xmin, Xmax, Ymin, Ymax）
        
        self.target_move_step = setting['target_move_step'] # 目标移动步长

        '''相机、目标、障碍物位置矩阵'''
        self.cameras = np.zeros((self.num_cameras, 3), dtype=int) # 相机矩阵[x, y, theta]
        self.targets = np.zeros((self.num_targets, 2), dtype=int) # 目标矩阵[x, y]
        if self.obstacle_flag == 1: # 有障碍物
            self.num_obstacles = np.random.randint(self.num_cameras-1, self.num_cameras) # 随机障碍物数量范围
            self.obstacle_radiusRange = setting['obstacle_radiusRange'] # 障碍物半径范围
            self.obstacles = np.zeros((self.num_obstacles, 3), dtype=int)# 障碍物矩阵[x, y, r]

        '''动作空间'''
        self.camera_action_space = spaces.MultiDiscrete([3, 3]) # 相机动作空间（执行者），旋转和移动
        # self.camera_coordinator_action_space = [spaces.Discrete(2) for i in range(self.num_cameras * self.num_targets)] # 相机高级动作空间（协调者），是否选择目标
        # self.target_action_space = spaces.Box(low=np.array([0, -180]), high=np.array([5, 180]), dtype=int) # 目标动作空间, 位移距离0-5，旋转角度-180-180

        '''观察空间是 n*m*2 的矩阵，n是相机数量，m是目标数量，2是距离和角度'''
        self.state_dim = 2 #观察空间维度
        if self.obstacle_flag == 0: # 无障碍物
            self.observation_space = np.zeros((self.num_cameras, self.num_targets, self.state_dim), dtype=np.float32) # 可见的话是（i, q, d_iq, alpha_iq） 不可见是(0, 0, 0, 0) 
        else: # 有障碍物,暂时没写
            self.observation_space = np.zeros((self.num_cameras, self.num_targets+self.num_obstacles, self.state_dim), dtype=np.float32)

        '''相机与目标之间的可见性关系矩阵'''
        self.visibility_matrix = np.zeros((self.num_cameras, self.num_targets), dtype=np.int32) # 1表示可见，0表示不可见

        '''吸引势能矩阵（无障碍物情况下的）'''
        self.attract_potential = np.zeros((self.num_cameras, self.num_targets), dtype=np.float32) 

        '''斥力势能矩阵(无障碍物情况下的)'''
        self.repel_potential = np.zeros((self.num_cameras, self.num_cameras), dtype=np.float32) # 斥力势能矩阵

        '''相机势能向量'''
        self.camera_potential = np.zeros(self.num_cameras, dtype=np.float32) # 相机势能向量

        self.rewards = np.zeros(self.num_cameras, dtype=np.float32) # 奖励
        self.reward_mode = 'hybrid' # 奖励模式
        # self.reward_manager = RewardFunction(mode=self.reward_mode) # 奖励函数管理器
        self.reward_manager = Reward_Cover_Only() # 奖励函数管理器, 目前只考虑覆盖率奖励
        '''初始化环境'''
        self.reset()

    '''-----------上面是init()---------------'''

    def reset(self):
        '''重置环境状态，包括相机、目标、障碍物的位置，每个新回合都会调用这个方法'''
        self.num_episodes += 1 # episode数量计数器
        self.steps = 0 # 步数计数器
        self.rewards = np.zeros(self.num_cameras, dtype=np.float32) # 奖励
        
        # self.state = self.observation_space # 初始化状态
        
        # ----初始化相机----

        # for i in range(self.num_cameras):
        #     '''
        #     随机初始化相机位置和角度
        #     相机位置初始化在区域边界
        #     角度初始化:扇形覆盖区域能够保证覆盖到矩形区域内部
        #     '''
            # edge = np.random.choice(['bottom', 'right', 'top', 'left'])
            # if edge == 'bottom':
            #     x = np.random.randint(self.area[0], self.area[1]+1)
            #     y = self.area[2]
            #     theta = np.random.uniform(0, 180)
            #     theta += np.random.uniform(-self.visual_angle/2, self.visual_angle/2) # 在朝向矩形内部的基础方向上，加入±visual_angle/2 的扰动，使相机初始扇形随机分布
            # elif edge == 'right':
            #     x = self.area[1]
            #     y = np.random.randint(self.area[2], self.area[3]+1)
            #     theta = np.random.uniform(90, 180) if np.random.rand() < 0.5 else np.random.uniform(-180, -90)
            #     theta += np.random.uniform(-self.visual_angle/2, self.visual_angle/2)
            # elif edge == 'top':
            #     x = np.random.randint(self.area[0], self.area[1]+1)
            #     y = self.area[3]
            #     theta = np.random.uniform(-180, 0)
            #     theta += np.random.uniform(-self.visual_angle/2, self.visual_angle/2)
            # elif edge == 'left':
            #     x = self.area[0]
            #     y = np.random.randint(self.area[2], self.area[3]+1)
            #     theta = np.random.uniform(-90, 90)
            #     theta += np.random.uniform(-self.visual_angle/2, self.visual_angle/2)
            # self.cameras[i] = [x, y, theta]
        self.cameras = np.array([[-1200, -600, 45], [0, -600, 90], [1200, -600, 135], [1200, 600, -135], [0, 600, -90], [-1200, 600, -45]])

        # ----初始化目标----
        self.goal_agents = []
        for i in range(self.num_targets): # 随机初始化目标位置
            self.targets[i] = [np.random.randint(self.area[0], self.area[1]), np.random.randint(self.area[2], self.area[3])]
            goal = GoalNavAgent(i, self.area) # goal生成
            self.goal_agents.append(goal)  


        if self.obstacle_flag == 1: # 有障碍物的情况, 随机初始化障碍物位置
            for i in range(self.num_obstacles):
                self.obstacles[i] = [np.random.randint(self.area[0], self.area[1]), np.random.randint(self.area[2], self.area[3]), np.random.randint(self.obstacle_radiusRange[0], self.obstacle_radiusRange[1])]
        
        # 更新状态
        self.state = self.update_state() 
        return self.state
    
    def update_state(self):
        '''更新状态, 将全局状态和局部观测分开管理，全局状态是所有相机和目标的状态，局部观测是每个相机自己看到的目标情况'''
        obs_space = np.zeros_like(self.observation_space) # obs_space是全局状态
        for i in range(self.num_cameras):
            for j in range(self.num_targets):
                obs_space[i][j][0] = self.get_distance(self.cameras[i], self.targets[j]) # 计算距离
                obs_space[i][j][1] = self.get_angle(self.cameras[i], self.targets[j]) # 计算角度
                self.attract_potential[i][j] = calculate_attract_potential(self.cameras[i], self.targets[j], self.visual_distance, self.visual_angle) # 计算吸引势能
        
        for i in range(self.num_cameras): # 计算斥力势能
            for j in range(self.num_cameras):
                if i == j: 
                    self.repel_potential[i][j] = 0
                self.repel_potential[i][j] = calculate_repel_potential(self.cameras[i], self.cameras[j])
        
        for i in range(self.num_cameras): # 计算相机总势能
            self.camera_potential[i] = np.sum(self.attract_potential[i]) + np.sum(self.repel_potential[i])

        self.calculate_visible() # 计算可见性

        # obs_list是每个相机自己的观测 
        obs_list = []
        for i in range(self.num_cameras):
            camera_obs = []
            for j in range(self.num_targets):
                if self.visibility_matrix[i][j] == 1:
                    d = obs_space[i][j][0] # 距离
                    theta = obs_space[i][j][1] # 角度
                    camera_obs.append([d, theta])
            obs_list.append(camera_obs) # 每个相机的观测是一个列表，里面是可见目标的距离和角度
        self.state = {
            'global': obs_space,
            'obs': obs_list
        }
        return self.state

    def step(self, actions):
        '''执行动作, 先不考虑障碍物场景
        actions:联合动作
        '''
        self.steps += 1

        '''相机执行者动作'''
        for i in range(self.num_cameras):
            rotation_action, move_action = actions[i] // 3, actions[i] % 3 # 第i个相机的动作
            self.camera_rotate(i, rotation_action)
            self.camera_move(i, move_action)

        
        '''执行动作后，更新状态，计算奖励'''
        self.state = self.update_state()

        '''计算奖励'''
        self.calculate_rewards()

        '''目标移动'''
        self.target_move()


        '''终止条件'''
        done = False
        if self.steps >= self.max_steps:
            done = True
        
        info = {
            'coverage_rate': np.mean(np.any(self.visibility_matrix, axis=0)),
            'reward': self.rewards,
            'visible_targets': int(np.sum(np.any(self.visibility_matrix, axis=0))),
            'visible_counts': np.sum(self.visibility_matrix, axis=1), # 每个相机可见目标数量
        } # 这里可以加入一些调试信息
        # print(self.potential_matrix)
        # print(self.visibility_matrix)

        return self.state, self.rewards, done, info

    def render(self, save_path=None, return_rgb_array=False, ax=None):
        # 老的render方法
        # from .render import render
        # # print('Rendering environment...')
        # render(self.area, self.cameras, self.targets, self.visual_distance, self.visual_angle, self.steps, obstacle_pos=None, goal=None,
        #     comm_edges=None, save=False, visible=None)

        '''统一的渲染接口
        save_path:保存路径
        return_rgb_array:是否返回RGB数组,直接生成gif
        ax: 外部可选的matplotlib轴对象
        '''
        img = render(area=self.area,
                      camera_pos=self.cameras,
                      target_pos=self.targets,
                      visual_distance=self.visual_distance,
                      visual_angle=self.visual_angle,
                      episode_num=self.num_episodes,
                      step_num=self.steps,
                      obstacle_pos=self.obstacles if self.obstacle_flag == 1 else None,
                      visible=self.visibility_matrix,
                      save_path=save_path,
                      return_rgb_array=return_rgb_array,
                      ax=ax)
        if return_rgb_array:
            return img  # 返回内存帧

    def close(self):
        pass


    def set_cameras_position(self, camera_id, cameras_pos):
        '''设置相机位置, 传入的cameras_pos是一个列表[x, y, theta]'''
        self.cameras[camera_id] = cameras_pos
    
    def get_cameras_position(self, camera_id):
        '''获取相机位置'''
        return self.cameras[camera_id]
    
    def get_global_state(self):
        '''获取全局状态'''
        return self.state['global']

    def get_obs(self):
        '''获取每个相机的局部观察(供Actor使用)'''
        return self.state['obs']

    def get_angle(self, pos1, pos2): # pos1是相机位置，pos2是目标位置
        '''获取两个点之间的角度'''
        angle_x = np.degrees(np.arctan2(pos2[1]-pos1[1], pos2[0]-pos1[0])) # pos2[1]-pos[1]是delta_y, pos2[0]-pos1[0]是delta_x 转化为角度值 这是相机到目标向量与x轴的夹角
        delta_angle = angle_x - pos1[2] # pos1[2]是相机角度, 相机中轴和目标的夹角
        delta_angle = (delta_angle + 180) % 360 - 180 # 保证角度在-180到180之间
        return delta_angle
    
    def get_distance(self, pos1, pos2): #pos是相机或者目标
        '''获取两个点之间的距离'''
        return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
    
    def get_visibility(self, camera_id, target_id):
        '''获取相机和目标之间的可见性'''
        return self.visibility_matrix[camera_id][target_id]
    
    def target_move(self):
        if self.target_move_type == 'random':
            '''目前是随机移动'''
            for i in range(self.num_targets):
                '''这样生成方向来移动感觉有问题'''
                # angle = np.random.uniform(-180, 180) # 随机生成一个方向

                # # 计算目标位移
                # dx = self.target_move_step * math.cos(math.radians(angle)) 
                # dy = self.target_move_step * math.sin(math.radians(angle)) 
                
                # # 更新目标位置
                # self.targets[i][0] += dx
                # self.targets[i][1] += dy 

                '''HiT-MAC的移动方式'''
                self.targets[i][0] += np.random.randint(-self.target_move_step, self.target_move_step+1)
                self.targets[i][1] += np.random.randint(-self.target_move_step, self.target_move_step+1)

                # 边界处理
                self.targets[i][0] = max(min(self.targets[i][0], self.area[1]), self.area[0])
                self.targets[i][1] = max(min(self.targets[i][1], self.area[3]), self.area[2])
        elif self.target_move_type == 'goal':
            '''目标跟随agent移动'''
            for i in range(self.num_targets):
                goal = self.goal_agents[i]
                pose = np.array(self.targets[i])
                delta = goal.act(pose)  # 获取目标的移动向量

                # 更新目标位置
                self.targets[i][0] += delta[0]
                self.targets[i][1] += delta[1]

                # 边界处理
                self.targets[i][0] = max(min(self.targets[i][0], self.area[1]), self.area[0])
                self.targets[i][1] = max(min(self.targets[i][1], self.area[3]), self.area[2])
                



    def camera_rotate(self, camera_id, rotate_action):
        '''相机旋转动作执行'''
        if (rotate_action - 1) == 0: # 不旋转
            return
        elif (rotate_action - 1) == 1: # 逆时针旋转
            self.cameras[camera_id][2] += self.rotation_scale
        elif (rotate_action - 1) == -1: # 顺时针旋转
            self.cameras[camera_id][2] -= self.rotation_scale
        else:
            raise ValueError('Invalid rotate_action')
        
        while self.cameras[camera_id][2] > 180 or self.cameras[camera_id][2] < -180:
            if self.cameras[camera_id][2] > 180:
                self.cameras[camera_id][2] -= 360
            else:
                self.cameras[camera_id][2] += 360

    def camera_move(self, camera_id, move_action):
        '''相机移动动作执行'''
        if (move_action - 1) == 0: # 不移动 
            return 
        elif (move_action - 1) == 1: # 1:向前移动（逆时针: 下-右-上-左-下）
            direction = 1
        elif (move_action - 1) == -1: # -1:向后移动（顺时针: 下-左-上-右-下）
            direction = -1 

        x, y, theta = self.cameras[camera_id]
        xmin, xmax, ymin, ymax = self.area

        if y == ymin and xmin <= x < xmax: # 在下边
            if (x + direction * self.move_scale) > xmax: # 移动之后超出了边界框的情况
                d_broader = xmax - x # d_broader: 到边界的距离
                x = xmax
                y = ymin + (self.move_scale - d_broader)
            elif (x + direction * self.move_scale) < xmin: # direction自带方向，都是加号
                d_broader = x - xmin
                x = xmin
                y = ymin + (self.move_scale - d_broader)
            else:
                x += direction * self.move_scale
        
        elif x == xmax and ymin <= y < ymax: # 在右边
            if (y + direction * self.move_scale) > ymax:
                d_broader = ymax - y
                y = ymax
                x = xmax - (self.move_scale - d_broader)
            elif (y + direction * self.move_scale) < ymin:
                d_broader = y - ymin
                y = ymin
                x = xmax - (self.move_scale - d_broader)
            else:
                y += direction * self.move_scale
        elif y == ymax and xmin <= x < xmax: # 在上边
            if (x + direction * self.move_scale) > xmax:
                d_broader = xmax - x
                x = xmax
                y = ymax - (self.move_scale - d_broader)
            elif (x + direction * self.move_scale) < xmin:
                d_broader = x - xmin
                x = xmin
                y = ymax - (self.move_scale - d_broader)
            else:
                x += direction * self.move_scale 
        elif x == xmin and ymin <= y < ymax: # 在左边
            if (y + direction * self.move_scale) < ymin:
                d_broader = y - ymin
                y = ymin
                x = xmin + (self.move_scale - d_broader)
            elif (y + direction * self.move_scale) > ymax:
                d_broader = ymax - y
                y = ymax
                x = xmin + (self.move_scale - d_broader)
            else:
                y += direction * self.move_scale
        
        self.cameras[camera_id][:2] = [x, y]
    
    def calculate_visible(self):
        '''判断目标是否在相机的可视范围内'''
        for i in range(self.num_cameras):
            for j in range(self.num_targets):
                distance = self.get_distance(self.cameras[i], self.targets[j]) # 相机与目标距离
                if distance > self.visual_distance:
                    self.visibility_matrix[i][j] = 0 # 超出距离，不可见
                    continue
                angle = self.get_angle(self.cameras[i], self.targets[j]) # 相机与目标夹角
                if abs(angle) > self.visual_angle / 2:
                    self.visibility_matrix[i][j] = 0 # 超出角度，不可见
                    continue
                self.visibility_matrix[i][j] = 1
        
        
        
    def get_attract_potential(self, camera_id, target_id):
        '''获取目标对相机的吸引势能'''
        return self.attract_potential[camera_id][target_id]
    
    def get_repel_potential(self, camera_id1, camera_id2):
        '''获取两个相机之间的斥力势能'''
        return self.repel_potential[camera_id1][camera_id2]

    def get_camera_potential(self, camera_id):
        '''获取相机的势能'''
        return self.camera_potential[camera_id]
    
    # def get_local_reward(self, camera_id):
    #     '''获取某个相机的局部奖励'''
    #     return self.reward['local'][camera_id]

    def calculate_rewards(self):
        '''计算奖励'''
        # for i in range(self.num_cameras):
        #     if self.reward_mode == 'potential':
        #         self.rewards[i] = self.reward_manager.calculate_potential_reward(i, self.camera_potential)
        #     elif self.reward_mode == 'hybrid':
        #         self.rewards[i] = self.reward_manager.calculate_hybrid_reward(i, self.visibility_matrix)
        #     elif self.reward_mode == 'soft-transition':
        #         self.rewards[i] = self.reward_manager.calculate_soft_transition_reward(i, self.camera_potential, self.visibility_matrix)

        # 只考虑覆盖率
        self.rewards = self.reward_manager.calculate_reward(self.visibility_matrix)
    

        
class GoalNavAgent:
    '''目标导航智能体，是targets的运动模型'''

    def __init__(self, id, area, velocity_range=(50, 100), goal_list=None):
        '''
        初始化目标导航Agent
        id: 目标ID
        area: 区域范围 (xmin, xmax, ymin, ymax)
        velocity_range: 目标速度范围 (min_velocity, max_velocity)
        goal_list: 可选的预定义目标点序列(用于可控目标运动)
        '''
        self.id = id
        self.area = area
        self.velocity_low, self.velocity_high = velocity_range
        self.goal_list = goal_list

        self.smooth_prob = 0.5 # 新生成的目标平滑概率
        self.step_counter = 0 # 累计移动步数
        self.goal_id = 0 # 当前目标点ID，即更新过几次目标点
        self.max_len = 150 # 最大连续朝同一goal移动的步数
        self.pose_last = [[], []] # 最近两帧的位置，判断是否卡住

        # 初始化第一个目标点和速度
        self.goal = self.generate_goal(np.array([0, 0]), np.array([0, 0]), force_random=True) # 初始位置为(0, 0)，强制随机生成目标点
        self.velocity = self.sample_velocity()

    def sample_velocity(self):
        '''从速度范围内随机采样一个基准速度
        返回:float 基准速度(像素/步)
        '''
        return np.random.uniform(self.velocity_low, self.velocity_high) / 10.0

    def generate_goal(self, current_pose, current_goal, force_random=False, ):
        '''
        生成一个新的目标点(goal)
        返回: ndarray [x, y] 目标点坐标
        '''
        if self.goal_list and len(self.goal_list) != 0:
            # 如果有预定义目标点，按顺序取出，且循环使用
            index = self.goal_id % len(self.goal_list)
            goal = np.array(self.goal_list[index])
        else:
            # 否则在区域内随机生成
            if force_random or np.random.rand() > self.smooth_prob:
                # 20%概率完全随机生成goal
                x = np.random.uniform(self.area[0], self.area[1])
                y = np.random.uniform(self.area[2], self.area[3])
                goal = np.array([x, y])
            else:
                '''平滑过渡这块再考虑要不要，因为容易出现目标点在边界的情况'''
                # 平滑过渡生成goal：即在当前方向附近生成
                direction = current_goal - current_pose # 当前目标点到当前位置的方向向量
                norm = np.linalg.norm(direction)
                if norm == 0:
                    direction = np.array([1.0, 0.0]) # 如果当前位置和目标点重合，默认向右
                else:
                    direction = direction / norm
                
                # # 在当前方向附近 ±30度 生成新目标点
                # angle_offset = np.random.uniform(-30, 30) * np.pi /180 # 在当前方向附近±30度内随机生成
                # cos_a, sin_a = np.cos(angle_offset), np.sin(angle_offset)
                # rotated_dir = np.array([cos_a * direction[0] - sin_a * direction[1],
                #                         sin_a * direction[0] + cos_a * direction[1]])
                # 计算到区域中心的“回拉向量”
                area_center = np.array([
                    (self.area[0] + self.area[1]) / 2,
                    (self.area[2] + self.area[3]) / 2
                ])
                inward_vector = area_center - current_pose
                inward_vector = inward_vector / (np.linalg.norm(inward_vector)+1e-8)

                # 融合回拉向量，让新方向偏向区域中心
                direction = 0.7 * direction + 0.3 * inward_vector
                direction = direction / np.linalg.norm(direction)

                
                distance = np.random.uniform(200, 400) # 随机生成目标点距离当前位置的距离
                goal = current_pose + distance * direction # 生成新目标点

                # 边界处理，确保目标点在区域内  
                goal[0] = np.clip(goal[0], self.area[0], self.area[1])
                goal[1] = np.clip(goal[1], self.area[2], self.area[3])

        self.goal_id += 1 # 更新目标点ID
        return goal

    def check_reach(self, goal, now):
        '''
        判断是否到达goal
        goal: ndarray [x, y] goal坐标
        now: ndarray [x, y] target当前坐标
        '''
        distance = np.linalg.norm(goal - now)
        return distance < 5 # 目标点阈值半径
    
    def act(self, pose):
        '''
        根据当前位置pose，输出下一步的移动增量
        pose: ndarray [x, y] 当前目标位置
        返回: ndarray [dx, dy] 移动增量
        '''
        self.step_counter += 1
        # ----判断是否卡住（两帧之间的移动距离过小)----
        if len(self.pose_last[0]) == 0:
            # 初始化pose_last
            self.pose_last[0] = np.array(pose)
            self.pose_last[1] = np.array(pose)
            d_moved = 30 # 初始移动距离大于阈值，避免误触发
        else:
            # 计算最近两帧的移动距离
            d_moved = min(
                np.linalg.norm(np.array(self.pose_last[0]) - np.array(pose)),
                np.linalg.norm(np.array(self.pose_last[1]) - np.array(pose))
            )
            # 更新最近两帧位置
            # print(f'd_moved: {d_moved}, pose: {pose}')
            self.pose_last[0] = np.array(self.pose_last[1])
            self.pose_last[1] = np.array(pose)
        
        # ----判断是否需要生成新的目标点----
        if self.check_reach(self.goal, pose) or d_moved < 1 or self.step_counter > self.max_len:
            # 到达目标点或卡住或走到最大步数，生成新目标点
            self.goal = self.generate_goal(np.array(pose), self.goal)
            self.velocity = self.sample_velocity()
            self.step_counter = 0 # 重置步数计数器
        
        # ----计算移动增量----
        direction = self.goal - pose
        norm = np.linalg.norm(direction)
        if norm == 0:
            # 如果当前位置和目标点重合，返回零增量
            return np.array([0.0, 0.0]) # 防止除0
        else:
            # 计算单位方向向量并乘以速度
            direction = direction / norm 
        
        # ----加入速度扰动，增加灵活性----
        real_velocity = self.velocity * (1 + 0.2 * np.random.random()) # 速度扰动范围在[0.8, 1.2]之间
        
        # ----计算下一步增量----
        delta = direction * real_velocity # 计算移动增量
        return delta
    
    def reset(self):
        '''重置目标Agent状态'''
        self.step_counter = 0
        self.goal_id = 0
        self.pose_last = [[], []]
        self.goal = self.generate_goal(np.array([0, 0]), np.array([0, 0], force_random=True)) # 初始位置为(0, 0)，强制随机生成目标点
        self.velocity = self.sample_velocity()
    
