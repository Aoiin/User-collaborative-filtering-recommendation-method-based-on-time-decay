import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("micro_video/interactions.csv")
df.columns = [' ', 'user_id', 'item_id', 'time']
df = df.drop(columns=[' '])

# 时间格式转换
df['time'] = pd.to_datetime(df['time'])

# =============================
# 1. 提取日期
# =============================
df['date'] = df['time'].dt.date

# =============================
# 2. 按天统计观看总次数
# =============================
daily_watch_cnt = (
    df.groupby('date')
      .size()
      .reset_index(name='view_count')
)

# =============================
# 3. 画折线图
# =============================
plt.figure()
plt.plot(daily_watch_cnt['date'], daily_watch_cnt['view_count'], marker='o')
plt.xlabel("Date")
plt.ylabel("Number of Views")
plt.title("Daily Video Views")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
