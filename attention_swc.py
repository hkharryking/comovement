from torch import nn
import torch

class Attention_SWC(nn.Module):
    """
    Attention model with sliding window correlation
    """

    def get_code(self):
        return 'SWC'

    def __init__(self,window):
        super(Attention_SWC, self).__init__()
        self.window_size=window*3-2 # practical window size, keep in accordance with the other models

    def sliding_window(self,x, step_size=1):
        # unfold dimension to make the sliding window
        return x.unfold(2, self.window_size, step_size)

    def Pearson_correlation(self,x):
        result=[]
        for xs in x:
            line=[]
            for i in range(0,x.shape[2]):
                x1 = xs[ 0, i, :]
                x2 = xs[ 1, i, :]
                vx1 = x1 - torch.mean(x1)
                vx2 = x2 - torch.mean(x2)
                corr = torch.sum(vx1 * vx2) / (torch.sqrt(torch.sum(torch.pow(vx1,2))) * torch.sqrt(torch.sum(torch.pow(vx2,2))))
                line.append(corr)
            result.append(torch.stack(line, dim=1))
        result=torch.stack(result, dim=1).squeeze()
        return result

    def attention_generation(self,x):
        #  current value - average
        correlations=self.Pearson_correlation(x)
        average=correlations.sum(0)/correlations.shape[0]
        result=correlations-average
        return result * (result > 0).float()

    '''
    returns: attention, sectors score
    '''
    def forward(self, x):
        s_out = self.sliding_window(x)
        attentions=self.attention_generation(s_out)
        return attentions,None
