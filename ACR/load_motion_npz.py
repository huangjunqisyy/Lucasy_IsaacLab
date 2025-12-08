import numpy as np

# --- 1. 修改这里 ---
# 替换成你下载的数据集文件的真实路径
file_path = '/home/lucas/whole_body_tracking/walking.npz' 
# ---------------

try:
    # 2. 加载 .npz 文件
    # 这会返回一个 NpzFile 对象，它像一个字典
    data = np.load(file_path)

    # 3. 查看该文件中存储了哪些数组（即所有的“键”）
    print(f"--- 正在检查文件: {file_path} ---")
    print("\n文件中包含的 键 (Keys):")
    print(data.files)

    # 4. 遍历所有的键，并打印出对应数组的维度 (shape)
    print("\n--- 各个数组的维度信息 (Shape) ---")
    if not data.files:
        print("这个 .npz 文件是空的。")
    
    for key in data.files:
        # 通过键来访问特定的数组
        array = data[key]
        
        # .shape 属性会告诉你这个数组的维度
        print(f"  键 (Key): '{key}'")
        print(f"  维度 (Shape): {array.shape}")
        print(f"  数据类型 (Dtype): {array.dtype}")
        print("-" * 20)

    # 5. （可选）如果你想查看某个具体键（比如 'poses'）的内容
    # if 'poses' in data.files:
    #     print(f"\n'poses' 数组的前3个元素:\n{data['poses'][:3]}")

    # 6. 关闭文件
    data.close()

except FileNotFoundError:
    print(f"错误：文件未找到，请检查路径: {file_path}")
except Exception as e:
    print(f"加载文件时发生错误: {e}")