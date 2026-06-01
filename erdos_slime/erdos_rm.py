# # 已知较好的初始解，C5 ≈ 0.4095（比均匀解好很多）
# # 这是一个简单的阶跃函数构造
# INITIAL_CODE = '''
# def run(seed=42, budget_s=60, **kwargs):
#     import numpy as np
#     n = 200
#     h = np.zeros(n)
#     # 简单阶跃函数：[0,0.5]区间为1，[0.5,1]区间为0，周期为1
#     for i in range(n):
#         x = 2.0 * i / n
#         if x < 0.5 or (1.0 <= x < 1.5):
#             h[i] = 1.0
#     # 归一化使 sum = n/2
#     s = np.sum(h)
#     if s > 0:
#         h = h * (n / 2.0 / s)
#     h = np.clip(h, 0, 1)
#     dx = 2.0 / n
#     c5 = float(np.max(np.correlate(h, 1.0 - h, mode="full") * dx))
#     return h, c5, n
# '''

# from __future__ import annotations
# import asyncio
# import sys

# sys.path.insert(0, '/root/workspace/erdos/erdos_slime')

# from erdos_reward import compute_reward_from_code


# def _extract_code(text: str) -> str:
#     if '```python' in text:
#         return text.split('```python')[1].split('```')[0].strip()
#     if '```' in text:
#         return text.split('```')[1].split('```')[0].strip()
#     return text.strip()


# def _compute(sample) -> float:
#     code = _extract_code(sample.response)
#     if not code:
#         return 0.0

#     info = compute_reward_from_code(code)

#     if not info.get('success'):
#         return 0.0

#     # Standard reward: 1 / (eps + C5), lower C5 = higher reward
#     return float(info.get('reward', 0.0))


# async def reward(args, sample):
#     loop = asyncio.get_event_loop()
#     r = await loop.run_in_executor(None, _compute, sample)
#     return r
from __future__ import annotations
import asyncio
import sys
import os
import json
import fcntl  # 新增：用于进程间文件锁

sys.path.insert(0, '/root/workspace/erdos/erdos_slime')

from erdos_reward import compute_reward_from_code

BUFFER_PATH = '/root/workspace/erdos/data/shared_buffer.json'
LOCK_PATH = '/root/workspace/erdos/data/shared_buffer.lock'
MAXSIZE = 64

def _load_buf():
    try:
        with open(BUFFER_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def _update_buf_safe(new_entry):
    """带文件锁的线程/进程安全更新函数"""
    os.makedirs(os.path.dirname(BUFFER_PATH), exist_ok=True)
    
    # 获取文件排他锁，防止其他进程同时修改
    with open(LOCK_PATH, 'w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            # 1. 拿到锁之后，必须重新读取最新数据！
            buf = _load_buf()
            
            # 2. 加入新成绩
            buf.append(new_entry)
            
            # 3. 排序并淘汰末位
            buf.sort(key=lambda x: x['c5'])
            if len(buf) > MAXSIZE:
                buf.pop()
                
            # 4. 安全写入
            with open(BUFFER_PATH, 'w') as f:
                json.dump(buf, f)
        finally:
            # 释放锁
            fcntl.flock(lock_file, fcntl.LOCK_UN)

def _extract_code(text: str) -> str:
    if '```python' in text:
        return text.split('```python')[1].split('```')[0].strip()
    if '```' in text:
        return text.split('```')[1].split('```')[0].strip()
    return text.strip()

def _compute(sample) -> float:
    response_text = getattr(sample, 'response', '') or getattr(sample, 'text', '')
    code = _extract_code(response_text)
    
    if not code:
        return 0.0

    # 这里的 load 只是为了获取初始值给沙盒用，读到稍微旧一点的没关系
    buf = _load_buf()
    init_h = buf[0]['h'] if buf else None

    # 运行沙盒测试
    info = compute_reward_from_code(code, initial_h_values=init_h)

    if not info.get('success'):
        return 0.0

    raw_c5 = info.get('c5_bound', 1.0)
    
    reward = float(info.get('reward', 0.0))
    if reward == 0.0:
        reward = 1.0 / (0.001 + raw_c5)

    # === 动态存盘：安全写入 ===
    if reward > 0:
        new_entry = {
            'code': code,
            'h': info.get('h_values', []),
            'c5': raw_c5,
            'reward': reward,
            'n': 1,
        }
        # 调用安全的写入函数，彻底杜绝数据被覆盖
        _update_buf_safe(new_entry)

    return float(reward)

async def reward(args, sample):
    loop = asyncio.get_event_loop()
    r = await loop.run_in_executor(None, _compute, sample)
    return r