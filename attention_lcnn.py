from torch import nn
from torch.autograd import Variable
from crnn_stock.attention_cnne import Attention_CNNE
import torch

HIDDEN_UNIT = 128
HIDDEN_LAYER = 2

class Attention_LCNN(Attention_CNNE):
    """
    non-weighted
    """
    def get_code(self):
        return 'LCNN'

    def __init__(self,feature_num,filters_num,batch_size,window,timestamps,sector_cardinality,d,da):
        super(Attention_LCNN, self).__init__(feature_num,filters_num,batch_size,window,timestamps,sector_cardinality,d,da)
        self.lstm = nn.LSTM(
                                input_size = 1,
                                hidden_size = HIDDEN_UNIT,     #  hidden unit
                                num_layers = HIDDEN_LAYER,
                                dropout=0.15,
                                batch_first=True
                )
        self.da=da
        self.W1=Variable(torch.randn(da,timestamps-window*3+3)).cuda()
        self.W2=Variable(torch.randn(da,1)).cuda()
        self.hidden = self.init_hidden()
        self.line = nn.Sequential(nn.Linear(timestamps - window * 3 + 3, sector_cardinality))

    '''
    returns: attention weights, Sector classification score
    '''
    def forward(self, x):
        c_out = self.conv(
            x.contiguous())  # batch (time stamps), channel(features) ,height(1 dimension),width(combinations)
        c_out = c_out.transpose(1, 2)
        l_out, self.hidden = self.lstm(c_out, self.hidden)
        l_out = l_out.sigmoid()

        attentions = []
        for seq in l_out:
            A = torch.mm(self.W2.transpose(0, 1), self.tanh(torch.mm(self.W1, seq)))
            attentions.append(self.softmax(torch.mm(seq, A.transpose(0, 1)).squeeze()))
        attentions = torch.stack(attentions, dim=1).squeeze()
        return attentions.transpose(0, 1), self.line(attentions.transpose(0, 1))


    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        h0 = Variable(torch.zeros(HIDDEN_LAYER, self.batch_size, HIDDEN_UNIT), requires_grad=True).cuda().float()
        c0 = Variable(torch.zeros(HIDDEN_LAYER, self.batch_size, HIDDEN_UNIT), requires_grad=True).cuda().float()
        h0 = torch.nn.init.xavier_uniform(h0)
        c0 = torch.nn.init.xavier_uniform(c0)
        return (h0,c0) #LSTM

