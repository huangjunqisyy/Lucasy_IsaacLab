"""
实时打印Franka机械臂基座到末端执行器的齐次变换矩阵
使用方法: ./isaaclab.sh -p /path/to/this/script.py
"""

import argparse
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="实时打印Franka末端执行器变换矩阵")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
import isaaclab.sim as sim_utils
from pxr import UsdGeom

def create_homogeneous_matrix(translation, rotation_matrix):
    """构建4x4齐次变换矩阵"""
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix  # 旋转部分
    transform[:3, 3] = translation       # 平移部分
    return transform

def get_world_transform(stage, prim_path):
    """获取USD prim的世界坐标系变换"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None, None
    
    xform = UsdGeom.Xformable(prim)
    world_transform = xform.ComputeLocalToWorldTransform(0.0)
    
    # 提取位置
    translation = world_transform.ExtractTranslation()
    translation = np.array([translation[0], translation[1], translation[2]])
    
    # 提取旋转矩阵
    rotation = world_transform.ExtractRotationMatrix()
    rotation_matrix = np.array([
        [rotation[0][0], rotation[0][1], rotation[0][2]],
        [rotation[1][0], rotation[1][1], rotation[1][2]],
        [rotation[2][0], rotation[2][1], rotation[2][2]]
    ])
    
    return translation, rotation_matrix

def main():
    try:
        # 获取USD Stage
        context = omni.usd.get_context()
        stage = context.get_stage()
        
        # 创建地面
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/Ground", ground_cfg)
        
        # 创建灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # 导入Franka机器人
        robot_cfg = sim_utils.UsdFileCfg(
            usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Robots/Franka/franka_alt_fingers.usd"
        )
        robot_prim_path = "/World/Franka"
        robot_cfg.func(robot_prim_path, robot_cfg)
        
        # 定义关键路径
        base_link_path = f"{robot_prim_path}/panda_link0"      # 基座链接
        ee_link_path = f"{robot_prim_path}/panda_hand"         # 末端执行器
        
        count = 0
        print_interval = 30  # 每30帧打印一次
        
        # 主循环
        while simulation_app.is_running():
            simulation_app.update()
            count += 1
            
            # 定期打印变换矩阵
            if count % print_interval == 0:
                # 获取基座和末端的世界坐标系变换
                base_pos, base_rot = get_world_transform(stage, base_link_path)
                ee_pos, ee_rot = get_world_transform(stage, ee_link_path)
                
                if base_pos is not None and ee_pos is not None:
                    # 构建世界坐标系变换矩阵
                    T_world_base = create_homogeneous_matrix(base_pos, base_rot)
                    T_world_ee = create_homogeneous_matrix(ee_pos, ee_rot)
                    
                    # 计算相对变换: T_base_ee = inv(T_world_base) * T_world_ee
                    T_base_ee = np.linalg.inv(T_world_base) @ T_world_ee
                    
                    # 打印结果
                    print("\n" + "-"*80)
                    print(f"帧数: {count}")
                    print("-"*80)
                    print("基座到末端执行器的齐次变换矩阵 (Base -> End-Effector):\n")
                    print("T_base_to_ee = ")
                    for i in range(4):
                        row = "  [" + ", ".join([f"{T_base_ee[i, j]:9.6f}" for j in range(4)]) + "]"
                        print(row)
                    
                    # 提取并打印位置
                    position = T_base_ee[:3, 3]
                    print(f"\n位置 (x, y, z): [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}] m")
                    print("-"*80)
        
    except KeyboardInterrupt:
        print("\n程序已停止")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()