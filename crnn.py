import torch
from torch import nn


class CRNN(nn.Module):
    """
    A mixed deep learning framework with Convolution and LSTM
    """

    def __init__(self,feature_num, filters_num, window, ticker_num, hidden_unit_num,hidden_layer_num,dropout_ratio):
        pass

    def weights_init(m):
        for m in m.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform(m.weight.data)
                torch.nn.init.xavier_uniform(m.bias.data)

    def init_hidden(self):
        pass

    def forward(self, x):
        # nn.Conv1d input format：batch x channel(time steps for time series) x height (2 time series) x width (feature numbers)
        # LSTM inputs: x (batch, time_step, input_size)
        # r_out (batch, time_step, hidden_size)

        c_out = self.conv(x.contiguous())  # batch (time stamps), channel(features) ,height(1 dimension),width(combinations)
        c_out = self.bn(c_out.contiguous())
        # c_out=self.pool(c_out.transpose(0,2))
        r_input = c_out.transpose(0, 1).transpose(1, 2)
        r_out, self.hidden = self.rnn(r_input, self.hidden)
        # r_out, self.hidden = self.rnn(r_input, self.hidden)
        # r_out=self.batchnormal(r_out)
        # input reshape to: (batch, steps, input size)
        # batch (time steps of time series, since the new x and y are one-to-one with the actual time steps),
        # time step (results from the convolution),
        # input size (edge numbers)
        outs = []  # save all predictions
        for time_step in range(r_out.size(1) - 1):  # calculate output for each time step
            outs.append(self.line(r_out[:, time_step, r_out.size(2) - 1].unsqueeze(0)))
        # outs=self.out(r_out[:,:,])
        return torch.stack(outs, dim=1)

    def classify_result(self, predicitons):
        r1 = predicitons.sigmoid()
        r0 = -predicitons.sigmoid()
        result = torch.stack((r1, r0), dim=1).view(-1, 2)
        return result


class Reward_loss(nn.Module):

    def differential(self, x):
        x = x.squeeze()
        x1 = x[1:]
        # x1=x[:,1:,:]
        # diff=(x1-x[:,:-1,:])/1.0/x1
        diff = (x1 - x[:-1]) / 1.0 / x1
        return diff

    def forward(self, input, target):
        diff_input = self.differential(input)
        diff_target = self.differential(target)
        return torch.sum(diff_target - diff_input.squeeze()).sigmoid()  # differential percentage loss
        # return torch.sum(self.alpha+target-input)

class Risk_loss(nn.Module):
    def forward(self, input):
        pass

