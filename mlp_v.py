import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import copy
#defines the network for value function

class v_approx(nn.Module):
    def __init__(self,n_input,n_hidden1,n_hidden2,n_hidden3,n_output,epoch,learning_rate) -> object:
        super(v_approx,self).__init__()
        self.hidden1 = torch.nn.Linear(n_input,n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1,n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2,n_hidden3)
        self.predict = torch.nn.Linear(n_hidden3,n_output)
        self.epo = epoch
        self.lr = learning_rate

    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)
        out = self.hidden3(out)
        out = F.relu(out)
        out = self.predict(out)
        return out
    
    def train_v(self,v,v_func):
        optimizer = torch.optim.Adam(v.parameters(), lr=self.lr)
        loss_func = torch.nn.MSELoss()
        for epoch in range(self.epo):
            for batch in v_func:
                x = torch.cat([a[0] for a in batch])
                y = torch.cat([a[1].view([1,1]) for a in batch])
                predic = v(x)
                loss = (loss_func(predic,y))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        return
    def train_v_func_inloop(self,v_func,v,threshold): 
        mse_test = 0
        mse_min = float("inf")
        e = 0
        random.shuffle(v_func)
        test = v_func[:int(0.2*len(v_func))]
        train = v_func[int(0.2*len(v_func)):]
        training_data = [train[a:a+512] for a in range(0,len(train)-len(train)%512+1,512)]
        if [] in training_data:
            training_data.remove([])
        while mse_min - mse_test >= threshold:
            if mse_min > mse_test and e != 0:
                mse_min = float(mse_test)
            v.train_v(v,training_data)
            x = torch.cat([a[0] for a in test])
            y = torch.cat([a[1].view([1,1]) for a in test])
            predic = v(x)
            mse_test = ((sum((predic-y)**2))/len(y))**0.5
            e += 1
            #print("RMSE",mse_test)
        #print("e",e)
        return 
