import numpy as np
import time 


def create_env(env_id, args, rank=-1):
    if 'MSMTC' in env_id:
        import MSMTC.DigitalPose2D as poseEnv
        
