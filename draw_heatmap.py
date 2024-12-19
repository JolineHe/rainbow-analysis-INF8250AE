import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取 CSV 文件

agents = ['dqn', 'ddqn', 'multistep_dqn', 'dueling_ddqn', 'distributional_dqn', 'noise_dqn', 'prioritized_ddqn','dp']
# agents = ['dp']

for agent in agents:
    file_path = f'figures/{agent}_mean_value_function.csv'
    df = pd.read_csv(file_path, index_col=0)
    df = df.T

    # 创建热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='coolwarm', cbar=True)

    # 添加标题和标签
    plt.title(f'Heatmap of {agent} Value Function')
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    plt.savefig(f"figures/heatmaps/{agent}_value_heatmap.png")

    # 显示热图
    plt.show()
