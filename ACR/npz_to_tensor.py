import torch
from tensordict import TensorDict
import numpy as np

# ----------------------- 你的函数（保持不变） -----------------------
def npz_to_obs_tensor_dict(
        npz_path: str,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32
    ) -> TensorDict:
        """
        从.npz文件提取关节/身体数据，按指定顺序拼接并封装为与obs（TensorDict）匹配的格式。
        
        关键映射关系（npz键 → obs键）：
        - body_lin_vel_w      → base_lin_vel（基础线速度）
        - body_ang_vel_w    → base_ang_vel（基础角速度）
        - joint_pos         → joint_pos（关节位置）
        - joint_vel         → joint_vel（关节速度）
        - body_pos_w        → body_pos_w（身体位置，全局坐标系）
        
        Args:
            npz_path: .npz文件路径（需包含上述5个键）
            device: 输出Tensor的设备（如torch.device("cuda:0")）
            dtype: 输出Tensor的数据类型（默认float32，与强化学习观测一致）
        
        Returns:
            obs_tensor_dict: 与观测匹配的TensorDict，键为上述obs键，值为对应torch.Tensor
        
        Raises:
            ValueError: 若.npz文件缺少目标键、数据维度不合法
            TypeError: 若.npz数据类型无法转换为torch.Tensor
        """
        # -------------------------- 步骤1：加载.npz文件并验证目标键 --------------------------
        try:
            with np.load(npz_path, allow_pickle=False) as npz_data:
                # 1.1 定义“npz键→obs键”的映射（含目标顺序）
                key_mapping = [
                    ("body_lin_vel_w", "base_lin_vel"),    # 1. 基础线速度
                    ("body_ang_vel_w", "base_ang_vel"),  # 2. 基础角速度
                    ("joint_pos", "joint_pos"),          # 3. 关节位置
                    ("joint_vel", "joint_vel"),          # 4. 关节速度
                    ("body_pos_w", "body_pos_w")         # 5. 身体位置（全局）
                ]
                
                # 1.2 验证所有目标键是否存在于.npz中
                missing_keys = [npz_key for npz_key, _ in key_mapping if npz_key not in npz_data]
                if missing_keys:
                    raise ValueError(
                        f".npz文件缺少以下必需键：{missing_keys}\n"
                        f".npz文件包含的所有键：{list(npz_data.keys())}"
                    )
                
                # 1.3 提取并转换每个键的数据（np.ndarray → torch.Tensor）
                data_dict = {}
                for npz_key, obs_key in key_mapping:
                    # 提取np数组
                    np_data = npz_data[npz_key]
                    
                    # 验证数据维度（至少2维：[时间步/环境数, 特征数]，避免1维数据）
                    if len(np_data.shape) < 2:
                        raise ValueError(
                            f"键{npz_key}的数据维度不合法：{np_data.shape}\n"
                            "要求至少2维（如[num_timesteps, feature_dim]或[num_envs, feature_dim]）"
                        )
                    
                    # 转换为torch.Tensor并移动到目标设备
                    try:
                        tensor_data = torch.tensor(np_data, dtype=dtype, device=device)
                    except TypeError as e:
                        raise TypeError(f"键{npz_key}的数据无法转换为torch.Tensor：{str(e)}")
                    
                    # 存储到字典（用obs键命名）
                    data_dict[obs_key] = tensor_data
            
            # -------------------------- 步骤2：封装为TensorDict（与obs格式匹配） --------------------------
            # TensorDict的键与obs完全一致，值为对应Tensor（维度、设备、 dtype均对齐）
            obs_tensor_dict = TensorDict(
                source=data_dict,
                batch_size=data_dict["base_lin_vel"].shape[:-1],  # 批量维度（如[num_timesteps]或[num_envs]）
                device=device
            )
            
            # -------------------------- 步骤3：打印日志（验证结果） --------------------------
            print("=" * 80)
            print(f".npz文件 {npz_path} 转换完成，观测TensorDict信息：")
            for obs_key, tensor in obs_tensor_dict.items():
                print(f"  键: {obs_key:15s} | 形状: {tensor.shape:20s} | 设备: {tensor.device} |  dtype: {tensor.dtype}")
            print("=" * 80)
            
            return obs_tensor_dict

        except FileNotFoundError:
            raise FileNotFoundError(f".npz文件未找到：{npz_path}")
        except Exception as e:
            raise RuntimeError(f"转换失败：{str(e)}") from e

# ----------------------- 验证程序 -----------------------
def main():
    # --- 请在这里修改你的配置 ---
    npz_file_path = "/home/lucas/isaac-sim/IsaacLab/beyondAMP/data/demo/motion.npz"  # 【重要】替换成你的测试.npz文件路径
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 自动选择设备 (CPU/GPU)
    dtype = torch.float32  # 数据类型，通常无需修改
    
    # 如果你知道每个键的预期维度，可以在这里设置，程序会进行验证
    # 格式: {"obs_key": (expected_batch_dim, expected_feature_dim)}
    # 例如: expected_shapes = {"base_lin_vel": (1000, 3), "joint_pos": (1000, 24)}
    expected_shapes = {} 
    # ---------------------------

    print(f"--- 开始验证 npz_to_obs_tensor_dict 函数 ---")
    print(f"测试文件路径: {npz_file_path}")
    print(f"目标设备: {device}")
    print(f"数据类型: {dtype}")
    if expected_shapes:
        print(f"预期维度: {expected_shapes}")
    print("-" * 50)

    # 调用函数进行转换
    try:
        obs_tensor_dict = npz_to_obs_tensor_dict(
            npz_path=npz_file_path,
            device=device,
            dtype=dtype
        )
        print("转换成功！")
    except Exception as e:
        print(f"\n❌ 转换失败！错误信息：")
        print(f"  {str(e)}")
        return

    # 如果提供了预期维度，则进行验证
    if expected_shapes:
        print("\n--- 开始验证维度 ---")
        all_shapes_correct = True
        for key, expected_shape in expected_shapes.items():
            if key in obs_tensor_dict:
                actual_shape = obs_tensor_dict[key].shape
                if actual_shape == expected_shape:
                    print(f"✅ 维度验证成功: '{key}' - 预期: {expected_shape}, 实际: {actual_shape}")
                else:
                    print(f"❌ 维度验证失败: '{key}' - 预期: {expected_shape}, 实际: {actual_shape}")
                    all_shapes_correct = False
            else:
                print(f"❌ 键 '{key}' 不在转换结果中。")
                all_shapes_correct = False
        
        if all_shapes_correct:
            print("🎉 所有维度验证通过！")
        else:
            print("❌ 部分维度验证失败，请检查你的预期或数据文件。")
    
    print("\n--- 验证结束 ---")

if __name__ == "__main__":
    main()