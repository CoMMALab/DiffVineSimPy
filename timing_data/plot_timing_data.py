import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import product
from pathlib import Path

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
max_bodies = [10, 20, 30, 40, 60, 80, 100, 120]
df = pd.concat(
    [
        pd.read_csv(f'timing_{bs}_{mb}.csv') for bs,
        mb in product(batch_sizes, max_bodies) if Path(f'timing_{bs}_{mb}.csv').exists()
        ]
    )

# df['num_bodies'] = df['num_bodies'].transform(lambda x: float(x[7:-1]))
# df['num_bodies'] = df['num_bodies'].round(0)


sns.set_context("paper", font_scale = 1.5)
sns.set_style("whitegrid")

height = 4
width = 8

ax = sns.catplot(
    x = 'max_bodies',
    y = 'time',
    hue = 'batch_size',
    palette = "muted",
    data = df,
    kind = 'bar',
    height = height,
    aspect = width / height
    )

sns.despine()
plt.yscale('log')
plt.savefig('time_per_iter.png')

df['time'] = df['time'] / df['batch_size']

sns.set_context("paper", font_scale = 1.5)
sns.set_style("whitegrid")

height = 4
width = 8

ax = sns.catplot(
    x = 'max_bodies',
    y = 'time',
    hue = 'batch_size',
    palette = "muted",
    data = df,
    kind = 'bar',
    height = height,
    aspect = width / height
    )

sns.despine()
plt.yscale('log')
plt.savefig('time_per_iter_per_robot.png')
