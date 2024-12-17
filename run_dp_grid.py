from envs.grid_world import GridWorld
import numpy as np
import pandas as pd 
from arguments import args           


env = GridWorld()
env_size = env.env_size
action_space = env.action_space
target_state = env.target_state
forbidden_states = env.forbidden_states
num_states = env.num_states
gamma = 0.9


value_function = np.zeros(env_size)  
policy = np.zeros(env_size, dtype=int)  
# policy_matrix=np.random.rand(num_states,len(action_space))

max_iterations = 1000  
theta = 1e-4  

if __name__ == "__main__":    
    state = env.reset() 
    env.render()
    for t in range(max_iterations):
        delta = 0  
        new_value_function = value_function.copy()

        for x in range(env_size[0]):
            for y in range(env_size[1]):
                state = (x, y)
                env.set_state(state)
                if state == target_state or state in forbidden_states:
                    continue

                action_values = []
                for _, action in enumerate(action_space):
                    state = (x, y)
                    env.set_state(state)
                    next_state, reward, done, _ = env.step(action)
                    print(f"interation: {t}, state:{state+(np.array([1,1]))}, Action: {action}, next State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")

                    action_value = reward + gamma * value_function[next_state]
                    action_values.append(action_value)

                best_action_value = max(action_values)
                new_value_function[state] = best_action_value

                policy[state] = np.argmax(action_values)
                delta = max(delta, abs(value_function[state] - best_action_value))

        value_function = new_value_function
        if delta < theta:
            break


print("Optimal Policy:")
print(policy)
# env.add_policy(policy)
env.add_state_values_matrix(value_function)
env.render(animation_interval=2)

value_function_df = pd.DataFrame(value_function, 
                                 columns=[f"Col {i}" for i in range(env_size[1])],
                                 index=[f"Row {i}" for i in range(env_size[0])])
print("Optimal Value Function Table:")
print(value_function_df)

input('press Enter to continue')

start = (7,0)
print(f"start from state:{start+(np.array([1,1]))}")
env.set_state(start)
env.render()
done = False
next_state = start
total_reward=0
while(not done):
    ply = policy[next_state]
    action = action_space[ply]
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    print(f"Action: {action}, next State: {next_state+(np.array([1,1]))}, Reward: {reward}, total reward:{total_reward}, Done: {done}")
    env.render()


input('press Enter to quit')
