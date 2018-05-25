from torch import nn
from torch.autograd import Variable
import torch

class Attention_CNNE(nn.Module):
    """
    A deep learning framework with Convolution Neural Network
    """

    def get_code(self):
        return 'CNNE'

    def __init__(self,feature_num,filters_num,batch_size,window,timestamps,sector_cardinality,d,da,):
        super(Attention_CNNE, self).__init__()
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
        self.line=nn.Sequential(nn.Linear(d, sector_cardinality))
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
        em_seq = []
        for seq in c_out:
           em_seq.append(self.embedding_generate(seq))
        em = torch.LongTensor(em_seq).cuda()
        e_out = self.embedding(Variable(em))

        result=[]
        attentions=[]
        for i in range(0,e_out.shape[0]):
           out=e_out[i]
           A=torch.mm(self.W2.transpose(0,1),self.tanh(torch.mm(self.W1, out)))
           result.append(A)
           attentions.append(self.softmax(torch.mm(out, A.transpose(0, 1)).squeeze()))
        result=torch.stack(result, dim=1).squeeze()
        attentions = torch.stack(attentions, dim=1).squeeze()
        return attentions.transpose(0,1),self.line(result)

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