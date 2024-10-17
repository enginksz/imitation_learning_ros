import torch
import os
import matplotlib.pyplot as plt

load_path = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/logs/metrics.pth'
load_weights = torch.load(load_path)
print(f"len train_steps:{len(load_weights['train_steps'])}")
print(f"len train_returns:{len(load_weights['train_returns'])}")
print(f"len test_returns_normalized:{len(load_weights['test_returns_normalized'])}")
print(f"len update_steps:{len(load_weights['update_steps'])}")
print(f"len alphas:{len(load_weights['alphas'])}")
print("#######train_steps#######")
print(load_weights['train_steps'])
print("#######train_returns#######")
print(load_weights['train_returns'])
print("########test_returns_normalized######")
print(load_weights['test_returns_normalized'])
print("#########update_steps#####")
print(load_weights['update_steps'])
print("######alphas########")
print(load_weights['alphas'])
#print("######entropies########")
#print(load_weights['entropies'])
#print("#######Q_values#######")
#print(load_weights['Q_values'])
#net.load_state_dict(load_weights)
train_returns = []
for i in range(len(load_weights['train_returns'])):
    train_returns.append(load_weights['train_returns'][i][0])

#print(train_returns)
plt.plot(load_weights['train_steps'], train_returns, label="test")
plt.legend()
plt.show()