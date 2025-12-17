import pandas as pd
import numpy as np
from collections import defaultdict

import pandas as pd

df = pd.read_csv("micro_video/interactions.csv")
df.columns = [' ', 'user_id', 'item_id', 'time']

# 删除无用列
df = df.drop(columns=[' '])

# ===== 关键修改开始 =====
# 字符串时间 → datetime
df['time'] = pd.to_datetime(df['time'])


# 按时间排序
df = df.sort_values('time')

# 80% 作为训练集，20% 作为测试集（基于时间）
split_time = df['time'].quantile(0.8)
train_df = df[df['time'] <= split_time]
test_df  = df[df['time'] > split_time]
print(train_df)
print(test_df)

def build_user_item_time(df):
    user_item_time = defaultdict(dict)
    for _, row in df.iterrows():
        user_item_time[row['user_id']][row['item_id']] = row['time']
    return user_item_time

user_item_time = build_user_item_time(train_df)

def jaccard_similarity(items1, items2):
    if len(items1 | items2) == 0:
        return 0.0
    return len(items1 & items2) / len(items1 | items2)

def time_decay(interaction_time, current_time, lamb=0.8):
    """
    interaction_time, current_time: pandas.Timestamp
    """
    # Timedelta → 秒 → 天
    dt_days = (current_time - interaction_time).total_seconds() / 86400.0

    # 防止负数（理论上不会，但保险）
    dt_days = max(dt_days, 0.0)

    return 1.0 / ((1.0 + dt_days) ** lamb)



def recommend_time_ubcf(user_id, user_item_time, K=10, lamb=0.8):
    scores = defaultdict(float)

    # 目标用户已看过的物品
    target_items = user_item_time[user_id]
    target_item_set = set(target_items.keys())

    # 当前时间（取目标用户最近一次行为）
    current_time = max(target_items.values())

    for other_user, items in user_item_time.items():
        if other_user == user_id:
            continue

        # 用户相似度
        sim = jaccard_similarity(
            target_item_set,
            set(items.keys())
        )

        if sim == 0:
            continue

        # 对相似用户的物品进行时间加权累积
        for item, t in items.items():
            if item not in target_item_set:
                w = time_decay(t, current_time, lamb)
                scores[item] += sim * w

    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked_items[:K]]

def recall_at_k(recommended_items, ground_truth_items, K):
    if len(ground_truth_items) == 0:
        return 0.0
    hit = len(set(recommended_items[:K]) & set(ground_truth_items))
    return hit / len(ground_truth_items)

def ndcg_at_k(recommended_items, ground_truth_items, K):
    """
    recommended_items: 推荐列表（有序）
    ground_truth_items: 测试集真实物品集合
    """
    dcg = 0.0
    for i, item in enumerate(recommended_items[:K]):
        if item in ground_truth_items:
            dcg += 1.0 / np.log2(i + 2)  # i 从 0 开始，所以 +2

    # 理想 DCG
    ideal_hits = min(len(ground_truth_items), K)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0

def evaluate_model(test_df, user_item_time_train, recommend_func, K_list=[5,10,20]):
    recalls = {k: [] for k in K_list}
    ndcgs   = {k: [] for k in K_list}

    # 测试集中每个用户的真实物品
    test_user_items = defaultdict(set)
    for _, row in test_df.iterrows():
        test_user_items[row['user_id']].add(row['item_id'])

    for user, true_items in test_user_items.items():
        if user not in user_item_time_train:
            continue

        recs = recommend_func(user, max(K_list))

        for K in K_list:
            recalls[K].append(
                recall_at_k(recs, true_items, K)
            )
            ndcgs[K].append(
                ndcg_at_k(recs, true_items, K)
            )

    return {
        "Recall": {k: np.mean(recalls[k]) for k in K_list},
        "NDCG":   {k: np.mean(ndcgs[k]) for k in K_list}
    }

print("\n===== Time-aware User-based CF lamb=0.8 =====")
results = evaluate_model(
    test_df,
    user_item_time,
    lambda u, K: recommend_time_ubcf(u, user_item_time, K, lamb=0.8)
)

for k in results["Recall"]:
    print(f"Recall@{k}: {results['Recall'][k]:.4f}")
    print(f"NDCG@{k}:   {results['NDCG'][k]:.4f}")

# 随机选一个用户
test_user = list(user_item_time.keys())[0]

rec_items = recommend_time_ubcf(
    test_user,
    user_item_time,
    K=10,
    lamb=0.8
)

print("Recommended items:", rec_items)
