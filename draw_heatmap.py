import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

agents = ['dqn', 'ddqn', 'multistep_dqn', 'dueling_ddqn', 'distributional_dqn', 'noise_dqn', 'prioritized_ddqn','dp']
# agents = ['dp']

for agent in agents:
    file_path = f'figures/{agent}_mean_value_function.csv'
    df = pd.read_csv(file_path, index_col=0)
    df = df.T

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='coolwarm', cbar=True)

    plt.title(f'Heatmap of {agent} Value Function')
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    plt.savefig(f"figures/heatmaps/{agent}_value_heatmap.png")

    plt.show()
