from utils import cycle, flatten_list_dicts, lineplot
import os

metrics = dict(train_steps=[], train_returns=[], test_steps=[], test_returns=[], test_returns_normalized=[], update_steps=[], predicted_rewards=[], alphas=[], entropies=[], Q_values=[])

file_prefix = os.environ['HOME'] + 'imitation_learning_ros/src/imitation_learning/logs/'

metrics['test_steps']=[1000]
metrics['test_returns']=[[-286.71448810765094, -286.727883726082, -286.73972685180036, -285.5176626718759, -280.76745371244976, -220.81247768228252, -286.71990643990875, -287.92324863751617, -285.55796191756957, -284.3536779994785, -274.8395074059706, -283.16298123250647, -285.5364007919033, -283.20459956827807, -285.5452663665577, -212.71645153872817, -42.205404873037786, -286.7363467553357, -286.73403120487046, -287.9354633034134, -287.9236419571162, -286.7131194789468, -280.75858149328, -286.7279610524972, -280.82252600142294, -284.37332761299933, -281.98248854343433, -287.9431074668717, -286.7374512907021, -285.5533784377608]]

lineplot(metrics['test_steps'], metrics['test_returns'], filename=f"{file_prefix}test_returns", title=f"GAIL: test Test Returns")