import matplotlib.pyplot as plt
from pyds import MassFunction
import torch

class Abnormality_fusion:
    def __init__(self,window):
        self.abnormality_period={}
        self.window=window
        self.positive_evidence=None
        self.negative_evidence=None
        self.mass_stat=None

    def abnormality_output(self,train_period,alpha=0.01):
        spy_data=self.datatool.load_target(train_period)
        plt.plot(spy_data,label='SPY')
        plt.legend(loc='upper right')
        sorted_period=sorted(self.abnormality_period.items(), key=lambda item:item[1], reverse=True)
        n=len(self.abnormality_period)
        i=0
        for ((window_begin, window_end),value) in sorted_period:
            plt.axvspan(window_begin, window_end, alpha=0.3, color='yellow')
            if i<=int(n*alpha): i+=1
            else: break
        plt.savefig('abnormality_detection.jpg')
        plt.close()

    def DS_statistics(self,attentions,targets,cardinal_number,ratio=0.01):
        if self.mass_stat is None:
            self.mass_stat = torch.zeros(cardinal_number,3,attentions.shape[1])
        self.Mass=[]
        for i in range(0,len(attentions)):
            attention=attentions[i]
            sector_label=targets[i]
            ranked_attention=attention.sort(descending=True)[1]
            top_index = ranked_attention[0:int(len(attention) * ratio)]  # find top value attentions
            down_index = ranked_attention[len(ranked_attention)-int(len(attention) * ratio):len(ranked_attention)] # find down value attentions
            for index in top_index:
                index = index.data.cpu().numpy()[0]
                if (attention[index]>0).all():
                    self.mass_stat[sector_label.cpu().data.numpy()[0]][1][index] += 1
            for index in down_index:
                index = index.data.cpu().numpy()[0]
                self.mass_stat[sector_label.cpu().data.numpy()[0]][0][index] += 1



    def DS_fusion(self,observation_size):
        for i in range(0,self.mass_stat.shape[0]):
            for j in range(0, self.mass_stat.shape[2]):
                self.mass_stat[i][2][j] = observation_size-self.mass_stat[i][0][j]-self.mass_stat[i][1][j]
        for j in range(0, self.mass_stat.shape[2]):
            Mass = []
            for i in range(0, self.mass_stat.shape[0]):
                m={}
                m['0'] = self.mass_stat[i][0][j]
                m['1'] = self.mass_stat[i][1][j]
                m['01'] = self.mass_stat[i][2][j]  # ambiguous_evidence
                m=MassFunction(m)
                Mass.append(m)
            fusion_result=Mass[0]
            for i in range(1,len(Mass)):
                fusion_result&=Mass[i]
            if fusion_result['1']!=0 and fusion_result['1']>fusion_result['0']:
                attention_window = (j, j + self.window * 3 - 2)
                if attention_window not in self.abnormality_period:
                    self.abnormality_period[attention_window]=1
                else:
                    self.abnormality_period[attention_window]+=1


    def vote_statistics(self,attentions,ratio=0.01):
        for attention in attentions:
            top_index = attention.sort(descending=True)[1][0:int(len(attention)*ratio)]  # find top value attentions
            for index in top_index:
                index = index.data.cpu().numpy()[0]
                attention_window=(index,index+self.window*3-2)
                if attention_window not in self.abnormality_period:
                    self.abnormality_period[attention_window]=1
                else:
                    self.abnormality_period[attention_window]+=1

    def Abnormal_period(self):
        if len(self.abnormality_period)>0:
            return self.abnormality_period
        else:
            return None