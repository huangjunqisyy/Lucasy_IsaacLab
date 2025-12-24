# =========================================================================== #
# G1 Robot Body Names Configuration
# =========================================================================== #

# 1. Root Name (根节点)
# 用于计算线速度和角速度 (Body Linear/Angular Velocity)
# G1 的浮动基座通常叫 "pelvis"
g1_root_name = ["pelvis"] 

# 2. End-Effector Names (末端执行器)
# 仅用于计算 "body_pos_w" (局部位置)
# 对应双手和双脚的末端 Link
g1_ee_names = [
    "left_ankle_roll_link",   # 左脚
    "right_ankle_roll_link",  # 右脚
    "left_wrist_roll_link",   # 左手
    "right_wrist_roll_link",  # 右手
]

# 3. Key Body Names (关键肢体)
# 用于计算 "body_quat_w" (6D 局部旋转)
# 包含 Root 和全身主要关节对应的 Link。
# 这里我们选取了能够代表各个肢体朝向的关键 Link。
g1_key_body_names = [
    "pelvis",                 # 根节点
    
    # --- 躯干 (Torso) ---
    # 对应腰部关节之后的 Link
    "torso_link",             
    
    # --- 腿部 (Legs) ---
    # 大腿 (Thigh): 通常选取髋关节链的最后一个 Link，或者中间主要的 Link
    "left_hip_pitch_link",    
    "right_hip_pitch_link",
    # 小腿 (Shin/Shank)
    "left_knee_link",         
    "right_knee_link",
    # 脚部 (Foot)
    "left_ankle_roll_link",   
    "right_ankle_roll_link",
    
    # --- 手臂 (Arms) ---
    # 大臂 (Upper Arm): 选取肩关节链的代表 Link
    "left_shoulder_pitch_link", 
    "right_shoulder_pitch_link",
    # 小臂 (Forearm)
    "left_elbow_link",          
    "right_elbow_link",
    # 手部 (Hand)
    "left_wrist_roll_link",     
    "right_wrist_roll_link",
]

g1_anchor_name = ["torso_link"]