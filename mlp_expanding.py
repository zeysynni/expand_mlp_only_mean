import numpy as np
import torch
import torch.nn.functional as F
import copy
from copy import deepcopy


# defines the policy model which is a nn


class AdamOptim:
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta

    def update(self, t, w, dw):
        # dw, db are from current minibatch
        # momentum beta 1
        # *** weights *** #
        if t == 1:
            self.m_dw = [self.m_dw * torch.ones(dw[i].shape) for i in range(len(dw))]
        self.m_dw = [self.beta1 * self.m_dw[i] + (1 - self.beta1) * dw[i] for i in range(len(dw))]

        # rms beta 2
        # *** weights *** #
        if t == 1:
            self.v_dw = [self.v_dw * torch.ones(dw[i].shape) for i in range(len(dw))]
        self.v_dw = [self.beta2 * self.v_dw[i] + (1 - self.beta2) * (dw[i] ** 2) for i in range(len(dw))]

        # bias correction
        m_dw_corr = [self.m_dw[i] / (1 - self.beta1 ** t) for i in range(len(dw))]
        v_dw_corr = [self.v_dw[i] / (1 - self.beta2 ** t) for i in range(len(dw))]

        # update weights and biases
        w = [(w[i] + self.eta * (m_dw_corr[i] / (np.sqrt(v_dw_corr[i]) + self.epsilon))).requires_grad_(True)
             for i in range(len(w))]
        return w

    def update_for_activation(self, t, w, dw):
        # dw, db are from current minibatch
        # momentum beta 1
        # *** weights *** #
        # only a 3-dim tensor
        if t == 1:
            self.m_dw = self.m_dw * torch.ones(dw.shape)
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        # rms beta 2
        # *** weights *** #
        if t == 1:
            self.v_dw = self.v_dw * torch.ones(dw.shape)
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)
        # bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1 ** t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** t)
        # update weights and biases
        w = (w + self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))).requires_grad_(True)
        return w

    def if_converge(self, w_old, w_new, index_old={}, index={}):
        if index:
            diff_index = sum([torch.norm((index_old[key] - index[key]), p="fro") for key in index.keys()])
            diff_w = sum([torch.norm((w_old[i] - w_new[i]), p="fro") for i in range(len(w_new))])
            return diff_w + diff_index
        else:
            return sum([torch.norm((w_old[i] - w_new[i]), p="fro") for i in range(len(w_new))])


class MLP(object):
    def __init__(self, trajectories, rate, seed, size, variance=0.8):
        self.learning_rate = rate
        self.decay_adv = 0.97
        self.network_size = size  # size should be a list
        self.seed = seed
        self.traj = trajectories
        self.var = torch.tensor(variance)
        self.eta_node = 1e-03  # .01
        self.eta_layer = 1e-02  # 2e-01  #.2or.5
        self.max = 10.
        #self.layer_grad_adv = []
        #self.layer_grad = []
        #self.acc = 0

    def weights(self):
        torch.manual_seed(self.seed)
        weights = []
        for weight in self.network_size:
            # weights.append(
            #    (torch.FloatTensor(weight[0], weight[1]).uniform_(-1, 1) / np.sqrt(weight[0])).requires_grad_(True))
            weights.append((torch.randn(weight) / np.sqrt(weight[0])).requires_grad_(True))
        return weights

    def forward(self, x, weights, index):
        for w in enumerate(weights):
            if w[0] == 0:
                out = torch.mm(x, w[1])
                if str(w[0]) in index.keys():
                    try:
                        out = index[str(w[0])][0] * out + (index[str(w[0])][1] + index[str(w[0])][2] * out) / (
                                    1 + out ** 2)
                    except Exception:
                        import pdb;
                        pdb.set_trace()
                        print(w)
                    out = out / torch.sqrt(torch.tensor(sum([a ** 2 for a in index[str(w[0])]])))
                else:
                    out = torch.tanh(out)
            else:
                out = torch.mm(out, w[1])
                if str(w[0]) in index.keys():
                    out = index[str(w[0])][0] * out + (index[str(w[0])][1] + index[str(w[0])][2] * out) / (1 + out ** 2)
                    out = out / torch.sqrt(torch.tensor(sum([a ** 2 for a in index[str(w[0])]])))
                else:
                    out = torch.tanh(out)
        return out

    def reset_acc_grad(self):
        self.layer_grad_adv = []
        self.layer_grad = []
        self.acc = 0

    def accumulate(self):
        self.acc += 1

    def size_info(self, weights, nn, index, v, training_data, R, actions, variance):
        # compute ita value for one position
        fisher, gs = nn.F_and_g(training_data, actions, weights, variance, index, v, R)
        # gs = (torch.mean(gs, dim=1)).view([1, -1])
        gs = gs.view([1, -1])
        ita = torch.mm(gs, torch.linalg.lstsq(fisher, gs.T).solution)
        return ita

    def choose_node(self, weights, nn, size, thres, index, v, training_data, R, actions, variance, x=0):
        # if need to expand a layer size
        torch.manual_seed(self.seed)
        w = deepcopy(weights)
        ratio = []
        w_init_all = []  # input weights
        w_init_all_out = []  # output weights
        #fisher = [sum(self.layer_grad)][0] / self.acc
        #layer_grad_adv = [sum(self.layer_grad_adv)][0] / self.acc
        #ita_init = torch.mm(layer_grad_adv.T, torch.linalg.lstsq(fisher, layer_grad_adv).solution)+self.eta_node
        ita_init = self.size_info(weights, nn, index, v, training_data, R, actions, variance) + self.eta_node
        print("node ita", ita_init)
        for init in range(10):
            ita = []
            w_init = []
            w_init_out = []
            for pos in enumerate(size[:-1]):
                add_1 = (torch.FloatTensor(pos[1][0], 1).uniform_(-1, 1) / np.sqrt(pos[1][0]))
                add_2 = (torch.zeros([1, size[pos[0] + 1][1]]))
                weights[pos[0]] = (torch.cat((weights[pos[0]].requires_grad_(False), add_1), 1)).requires_grad_(True)
                weights[pos[0] + 1] = (torch.cat((weights[pos[0] + 1].requires_grad_(False), add_2),
                                                 0)).requires_grad_(True)
                ita.append(self.size_info(weights, nn, index, v, training_data, R, actions, variance))
                w_init.append(deepcopy(weights[pos[0]]))
                w_init_out.append(deepcopy(weights[pos[0] + 1]))
                weights = deepcopy(w)
            ratio.append([a / ita_init for a in ita])
            w_init_all.append(w_init)
            w_init_all_out.append(w_init_out)
        for i in enumerate(ratio):
            if x < max(i[1]):
                x = max(i[1])
                loc = i[0]  # which init
                index = i[1].index(x)  # which pos
        print("node radio to init", x)
        if self.max > x > thres:
            w[index] = w_init_all[loc][index]
            w[index + 1] = w_init_all_out[loc][index]
            size[index][1] += 1
            size[index + 1][0] += 1
            print("change size: ", size)
        else:
            print("no change in size")
        return w, size

    def choose_layer(self, weights, nn, size, thres, index, v, training_data, R, actions, variance, act_para, x=0):
        # if need to add a hidden layer
        torch.manual_seed(self.seed)
        w = deepcopy(weights)
        index_ori = deepcopy(index)
        ratio = []
        w_init_all = []  # input weights
        w_init_all_out = []  # output weights
        #layer_grad = [sum(self.layer_grad)][0] / self.acc  # tensor
        #fisher = torch.mm(layer_grad, layer_grad.T) / self.traj
        #fisher = [sum(self.layer_grad)][0] / self.acc
        #layer_grad_adv = [sum(self.layer_grad_adv)][0] / self.acc
        #layer_grad_adv = [sum(self.layer_grad_adv)][0] / self.acc
        #ita_init = torch.mm(layer_grad_adv.T, torch.linalg.lstsq(fisher, layer_grad_adv).solution)+self.eta_layer
        ita_init = self.size_info(weights, nn, index, v, training_data, R, actions, variance) + self.eta_layer
        print("layer initial ita", ita_init)
        for pos in enumerate(size):
            ita = []
            w_init = []
            w_init_out = []
            for init in range(10):
                # die 1. Matrix durch zwei ersetzen
                lp = (torch.FloatTensor(pos[1][0], pos[1][0]).uniform_(-1, 1) / np.sqrt(pos[1][0])).requires_grad_(True)
                lp_2 = torch.linalg.lstsq(lp.clone().detach(),
                                          weights[pos[0]].clone().detach()).solution.requires_grad_(True)
                weights[pos[0]] = lp
                weights.insert(pos[0] + 1, lp_2)
                # added layer position and initial activation function parameters
                # the index number is the hidden layer number
                index[str(pos[0])] = act_para
                print("index", index)
                print("act para", act_para)
                ita.append(self.size_info(weights, nn, index, v, training_data, R, actions, variance))
                w_init.append(deepcopy(lp))
                w_init_out.append(deepcopy(lp_2))
                weights = deepcopy(w)
                index = deepcopy(index_ori)
            ratio.append([a / ita_init for a in ita])
            print("len", len(weights))
            print("ita", ita)
            # append for each init
            w_init_all.append(w_init)
            w_init_all_out.append(w_init_out)
        # find out the biggest ratio & corres. pos & init
        for i in enumerate(ratio):
            mean = sum(i[1]) / len(i[1])  # mean of this pos
            if x < mean:
                x = mean
                weight_init = i[0]  # note this pos
        if self.max > x > thres:
            # if mean of ita of one pos is higher than the threshold
            layer = ratio[weight_init].index(max(ratio[weight_init]))  # note this init
            size.insert(weight_init, [size[weight_init][0], size[weight_init][0]])
            w[weight_init] = w_init_all[weight_init][layer]  # pos, init
            w.insert(weight_init + 1, w_init_all_out[weight_init][layer])
            # add activation func param
            # if the to be added hidden layer is at the beginning, later hidden layer index should be added by 1
            keys = [i for i in index.keys()]
            for i in range(len(keys)):
                if weight_init <= int(keys[i]):
                    index_ori[str(int(keys[i]) + 1)] = index_ori.pop(keys[i])
            index_ori[str(weight_init)] = act_para
            print("added hidden layer: ", weight_init)
        else:
            print("no change in hidden layer")
        print("layer ratio to initial", x)
        print("ratio", ratio)
        print("size", size)
        return w, size, index_ori

    def fisher_matrix(self, grads, gs, grad_act={}, gs_act={}):
        # grad with adv/grad wo. adv
        # sum/different traj stack together
        # list of tensors
        for i in range(len(gs)):
            gs[i] = torch.cat([a for a in gs[i]], 0)  # put every set of para of weights into one, for 1 traj
        gs = (torch.cat([a for a in gs], 1)).clone().detach()  # put each traj (col) next to each other
        if gs_act:
            gs = gs.T
            for key in gs_act.keys():
                # shape: traj amount, para amount
                gs_act[key] = torch.cat([a[0].view([1, - 1]) for a in gs_act[key]], 0)
                gs = torch.cat((gs, gs_act[key]), 1)
            gs = gs.T
        form = []
        # save cumulated grad for each weight matrix
        for i in range(len(grads)):
            form.append(grads[i].shape)
            # vectorize weights matrix
            grads[i] = grads[i].reshape(-1, 1)
        # combine all the vectors together
        dtheta_re = torch.cat(grads, 0)
        if grad_act:
            for key in grad_act.keys():
                grad_act[key] = grad_act[key] / self.traj  # len(gs.T)
                dtheta_re = torch.cat((dtheta_re, grad_act[key].view([-1, 1])), 0)
                form.append((grad_act[key].view([-1, 1])).shape)
        fisher = torch.mm(gs, gs.T) / self.traj  # normalize by traj
        return fisher, form, dtheta_re, gs

    def fisher_for_activation(self, grads, gs):
        # grad with adv/grad wo. adv
        for i in range(len(gs)):
            gs[i] = torch.cat([a for a in gs[i]], 0)
        gs = (torch.cat([a for a in gs], 1)).clone().detach()
        dtheta_re = grads
        fisher = torch.mm(gs, gs.T) / len(gs)
        return fisher, dtheta_re

    def cal_update(self, fisher, form, dtheta_re):
        update = torch.linalg.lstsq(fisher, dtheta_re).solution
        updates = []
        nr = 0
        for f in enumerate(form):
            if f[0] == 0:
                nr_new = f[1][0] * f[1][1]
                updates.append(update[:nr + nr_new].view(f[1]))
                nr += nr_new
            else:
                nr_new = f[1][0] * f[1][1]
                updates.append(update[nr:nr + nr_new].view(f[1]))
                nr += nr_new
        return updates

    def backward_ng2_fast(self, training_data, actions, R, weights, value, variance, index):
        # use ng
        # use sum of each episode, do average over averages
        # with each set of episodes, train the policy on it
        print("lr", self.learning_rate)
        converged = False
        epoch = 1
        adam = AdamOptim(eta=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        adam_activation = AdamOptim(eta=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        improve = 1
        while not converged:
            # average of grad over all episode
            # amout of grad for each episode
            # go through every episode,len is how many episodes we use
            w_old = copy.deepcopy(weights)
            index_old = copy.deepcopy(index)
            w_adv = []
            [w_adv.append(0) for i in weights]
            # grad with adv
            layer_para = {}
            for key in index.keys():
                layer_para[key] = 0
            grads = []
            # grad wo adv
            layer_grad = {}
            for key in index.keys():
                layer_grad[key] = []
            for epi in range(len(training_data)):
                s = torch.cat([a for a in training_data[epi]])
                s_next = torch.cat([s[torch.arange(s.size(0)) != 0], torch.tensor([[0, 0, 0, 0, 0]]).float()])
                s_value = value(s)
                s_next_value = value(s_next)
                r = torch.cat([torch.tensor(a).view([1, 1]) for a in R[epi]])
                a = torch.cat([a.view([1, 1]) for a in actions[epi]])
                adv = r + self.decay_adv * s_next_value - s_value
                var = torch.cat([torch.tensor(a).view([1, 1]) for a in variance[epi]])
                a4 = self.forward(s, weights, index)
                adv_grad_1 = (torch.tensor([float(b) for b in (a - a4) / (var ** 2)])).view([-1, 1])
                adv_grad_2 = adv_grad_1 * a4 * adv
                sum(adv_grad_2).backward()
                w_adv = [w_adv[i] + weights[i].grad for i in range(len(weights))]
                [weights[a].grad.zero_() for a in range(len(weights))]
                for key in index.keys():
                    layer_para[key] += index[key].grad
                    index[key].grad.zero_()
                a4 = self.forward(s, weights, index)
                sum(adv_grad_1 * a4).backward()
                grads.append([(copy.deepcopy(weights[a].grad)).reshape(-1, 1) for a in range(len(weights))])
                [weights[a].grad.zero_() for a in range(len(weights))]
                for key in index.keys():
                    layer_grad[key].append([(copy.deepcopy(index[key].grad)).view(-1, 1)])
                    index[key].grad.zero_()

            # average over trajectories
            grad_adv = [a / len(training_data) for a in w_adv]
            fisher, form, dtheta_re, gs = self.fisher_matrix(grad_adv, grads, layer_para, layer_grad)
            #self.layer_grad_adv.append(dtheta_re)
            # self.layer_grad.append(gs)
            #self.layer_grad.append(fisher)
            #self.accumulate()
            # fisher, form, dtheta_re, gs = self.fisher_matrix(grad_adv, grads)
            with torch.no_grad():
                # update list of matrix for weights
                updates = self.cal_update(fisher, form, dtheta_re)
                weights = adam.update(epoch, weights, updates)
                i = 0
                for key in index.keys():
                    # layer_para[key] = layer_para[key]/len(training_data)
                    # fisher matrix/grad_adv vector
                    # f, d_r = self.fisher_for_activation(layer_para[key], layer_grad[key])
                    # f = torch.eye(f.shape[0])
                    # update_activation = torch.linalg.lstsq(f, d_r).solution
                    # update the parameter for each hidden layer seperately
                    index[key] = adam_activation.update_for_activation(epoch, index[key],
                                                                       updates[len(weights) + i].view([-1]))
                    # index[key] = adam_activation.update_for_activation(epoch, index[key], update_activation)
                    i += 1
            current_change = adam.if_converge(w_old, weights, index_old, index)
            if 0.1 > current_change > improve:
                print("converged")
                break
            elif epoch > 200 or torch.norm(updates[-1], p="fro") < 0.000001:
                print("maximal iteration exceed")
                break
            else:
                improve = adam.if_converge(w_old, weights)
                epoch += 1
        return weights, index

    def F_and_g(self, training_data, actions, weights, variance, index, value, R):
        # use ng
        # use sum of each episode, do average over averages
        # with each set of episodes, train the policy on it
        grads = []
        w_adv = []
        [w_adv.append(0) for i in weights]
        layer_para = {}  #
        for key in index.keys():  #
            layer_para[key] = 0  #
        # grad wo adv
        layer_grad = {}  #
        for key in index.keys():  #
            layer_grad[key] = []  #
        for epi in range(len(training_data)):
            s = torch.cat([a for a in training_data[epi]])
            s_next = torch.cat([s[torch.arange(s.size(0)) != 0], torch.tensor([[0, 0, 0, 0, 0]]).float()])
            s_value = value(s)
            s_next_value = value(s_next)
            r = torch.cat([torch.tensor(a).view([1, 1]) for a in R[epi]])
            a = torch.cat([a.view([1, 1]) for a in actions[epi]])
            adv = r + self.decay_adv * s_next_value - s_value
            var = torch.cat([torch.tensor(a).view([1, 1]) for a in variance[epi]])
            a4 = self.forward(s, weights, index)
            adv_grad_1 = (torch.tensor([float(b) for b in (a - a4) / (var ** 2)])).view([-1, 1])
            adv_grad_2 = adv_grad_1 * a4 * adv
            sum(adv_grad_2).backward()
            w_adv = [w_adv[i] + weights[i].grad for i in range(len(weights))]
            [weights[a].grad.zero_() for a in range(len(weights))]
            for key in index.keys():  #
                layer_para[key] += index[key].grad  #
                index[key].grad.zero_()  #
            a4 = self.forward(s, weights, index)
            sum(adv_grad_1 * a4).backward()
            grads.append([(copy.deepcopy(weights[a].grad)).reshape(-1, 1) for a in range(len(weights))])
            [weights[a].grad.zero_() for a in range(len(weights))]
            for key in index.keys():  #
                layer_grad[key].append([(copy.deepcopy(index[key].grad)).view(-1, 1)])  #
                index[key].grad.zero_()  #
        grad_adv = [a / len(training_data) for a in w_adv]
        # fisher, form, dtheta_re, gs = self.fisher_matrix(grad_adv, grads)
        fisher, form, dtheta_re, gs = self.fisher_matrix(grad_adv, grads, layer_para, layer_grad)  #
        return fisher, dtheta_re
