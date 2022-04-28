import torch.nn as nn
import torch


class SumNet(nn.Module):
    def __init__(self):
        super(SumNet, self).__init__()

    def forward(self, q_values):
        return torch.sum(q_values, dim=2, keepdim=True)  

class RQNet(nn.Module):
    def __init__(self, args):
        super(RQNet, self).__init__()
        self.args = args
        self.estim = nn.Sequential(nn.Linear(2*args.n_agents, args.rnn_hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(args.rnn_hidden_dim, args.n_agents)
                                    )  


    def forward(self, q_values):
        q_values_aux = torch.mean(q_values, dim=1, keepdim=True)

        maxim = torch.max(q_values, dim=1, keepdim=True)[0]
        q_values_aux = torch.cat((q_values_aux, maxim), dim=2)
        
        q_values_aux = self.estim(q_values_aux)
        q_values_est = q_values + q_values_aux
        
        return q_values_est 