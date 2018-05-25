from torch import nn
from torch.autograd import Variable
import torch

class Attention_CNN(nn.Module):
    """
    A deep learning framework with Convolution Neural Network
    """

    def get_code(self):
        return 'CNN'

    def __init__(self,feature_num,filters_num,batch_size,window,timestamps,sector_cardinality,d,da,):
        super(Attention_CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=feature_num * 2,
                      out_channels=filters_num,
                      kernel_size=window,
                      stride=1,
                      padding=0),
            nn.MaxPool1d(kernel_size=window, stride=1),
            # nn.Conv1d(in_channels=filters_num,
            #           out_channels=int(filters_num/2),
            #           kernel_size=window,
            #           stride=1,
            #           padding=0),
            #nn.BatchNorm1d(int(filters_num/2)),
            nn.BatchNorm1d(filters_num),
            nn.Conv1d(in_channels=int(filters_num),
                      out_channels=1,
                      kernel_size=window,
                      stride=1,
                      padding=0),
            nn.ReLU(True)
        )
        self.embedding = nn.Embedding(100000, d)
        #TODO: automatic cuda cpu switch?
        self.W1=Variable(torch.randn(da,timestamps-window*3+3)).cuda()
        self.W2=Variable(torch.randn(da,1)).cuda()
        self.weights_init(self.conv)
        self.tanh=torch.nn.Tanh()
        #self.line=nn.Sequential(nn.Linear(d, sector_cardinality))
        self.line = nn.Sequential(nn.Linear(timestamps - window * 3 + 3, sector_cardinality))
        self.d=d
        self.batch_size=batch_size
        self.softmax = nn.Softmax(dim=0)

    '''
    returns: attention, Sector classification score
    '''
    def forward(self, x):
        c_out = self.conv(x.contiguous())  # batch (time stamps), channel(features) ,height(1 dimension),width(combinations)
        c_out = c_out.squeeze()
        c_out=c_out.sigmoid()
        return c_out,self.line(c_out)

    def embedding_generate(self,seq):
        em_seq=[]
        for i in range(0,len(seq)):
            t=seq[i]
            for j in range(0,self.d):
                threshold=float(j+1)/self.d
                if (t<=threshold).all():
                    em_seq.append(j)
                    break
        return em_seq

    def weights_init(self,m):
        for m in m.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform(m.weight.data)