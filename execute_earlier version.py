import numpy as np
import torch
from swingup_data import collect_data  # sample episode
from mlp_expanding import MLP
from swingup_seeresult import show, diagramm
from mlp_v import v_approx  # state value estimation
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore")
# a code which does the learning of policy
# only mean value nn
"""
import pdb; pdb.set_trace()
assert not torch.isnan(w[3]).any(), print
try:
    assert not ita == 0.
except Exception as error:
    print(error)
    print("gs", gs.shape)
    print("gs", gs)
    print("fisher", fisher.shape)
    print("fisher", fisher)
else:
    pass
"""


def if_reset(average, average_reward, thr, red=False, res=False, ft=False):
    global count, monitor, count_1, l_rate, nn
    # sometimes it stays long at the beginning for some reason
    if average > 400:
        if not ft:
            l_rate = min(l_rate, 0.00001)
            # l_rate = 0.2 * l_rate
            nn.learning_rate = l_rate
            ft = True
    if average > thr[0] or monitor == 1:
        monitor = 1
        if average < np.max(average_reward):
            count += 1
        else:
            count = 0
        if count > thr[1]:
            red = True
            count = 0
            count_1 += 1
        if count_1 > thr[2] and not average > 400:
            res = True
            count_1 = 0
    return red, res, ft


seed = 4
l_rate = 1
size = [[5, 7], [7, 1]]
threshold_node, threshold_layer = 2, 1.5
expanding_node, expanding_layer = False, False
act_para = torch.tensor((1., 0., 0.)).requires_grad_(True)
index = {}

start = 1
T = 300

tra = 200
lr_min = 0.000001
threshold = [0, 8, 3]  # begin to count/reduce/reset
decay = 0.7  # lr decay
extern_var = var_best = 0.995  # extern std, to be reduced during training
var_decay = 0  # std decay factor
decay_value = 0.97  # decay for calculating value function
avr_reward, best, worst, m, upper, lower, avr_var = [[] for x in range(7)]
data_collector = collect_data(tra, decay_value, seed)
nn = MLP(tra, l_rate, seed, size, variance=0.2)  # variance is std
torch.manual_seed(seed)
weights = nn.weights()
monitor, count, count_1 = 0, 0, 0
fine_tune, reset = False, False
if start > 1:
    weights, size, v, index, var_best = data_collector.load_w()
    show(weights, MLP(tra, l_rate, seed, size, variance=.668), index)
else:
    v = v_approx(5, 124, 264, 16, 1, 8, 0.0002)  # epo,lr
min_var = []

for epo in range(start, T):
    print(epo)
    if not reset and avr_reward != [] and max(avr_reward) < 410:
        var = extern_var ** var_decay
        var_decay += 1
    else:
        var_decay = var_best
        var = extern_var ** var_decay
    training_data, R, actions, mean, variance = data_collector.letsgobrandon1(weights, nn, index, var)
    print("extern std: ", var, "variance: ", nn.var + var)
    avr_reward, best, worst, avr, m, upper, lower = data_collector.save(R, tra, avr_reward, best, worst, m, upper,
                                                                        lower, avr_var, variance)
    if avr > 415:
        min_var.append(nn.var + var)
        show(weights, nn, index, repeat=5)
    print("average reward", avr)
    print(avr_reward)
    v_func = data_collector.train_data_v(training_data, R)
    v.train_v_func_inloop(v_func, v, 0.002)  # threshold
    print("used episodes", len(training_data))
    if avr > max(avr_reward):
        print("save policy")
        data_collector.save_weights(weights, v, size, index, var_best)
        var_best = deepcopy(var_decay)
        print(size)
    reduce, reset, fine_tune = if_reset(avr, avr_reward, threshold, ft=fine_tune)
    if reset:
        weights, size, v, index, var_best = data_collector.load_w()
        var_decay = var_best
        var = extern_var ** var_decay
        print("reset both")
    elif reduce:
        l_rate = max(decay * l_rate, lr_min)
        nn.learning_rate = l_rate
        print("decrease learning rate")
        old_size = deepcopy(size)
        if expanding_node and len(size) > 1 and avr < 410:
            weights, size = nn.choose_node(weights, nn, size, threshold_node, index, v, training_data, R, actions,
                                           variance)
        if expanding_layer and old_size == size and avr < 410:
            weights, size, index = nn.choose_layer(weights, nn, size, threshold_layer, index, v,
                                                   training_data, R, actions, variance, act_para)
    else:
        weights, index = nn.backward_ng2_fast(training_data, actions, R, weights, v, variance, index)
    print("best trajectory:", best[epo - start], "worst trajectory:", worst[epo - start], "size:", size,
          "index:", index)
print(max(avr_reward), "\n", avr_reward.index(max(avr_reward)) + 1)
if min_var:
    print("var: ", min(min_var))
diagramm(T, m, upper, lower)
show(weights, MLP(tra, l_rate, seed, size, variance=extern_var ** var_decay), index)
