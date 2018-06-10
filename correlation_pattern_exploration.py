from crnn_stock.data_util import Data_util
from crnn_stock.attention_factory import Attention_factory
from crnn_stock.pattern_dataset import  PatternDataset
from crnn_stock.abnormality_fusion import Abnormality_fusion
from crnn_stock.evaluator_factory import Evaluator_factory
from sklearn.model_selection import train_test_split
from crnn_stock.metrics import Metrics

import torch
from torch.autograd import Variable
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np
import time

# Hyper Parameters
WINDOW = 10      # WINDOW size for the time series. based on the transaction days 5, 10, 15...
FEATURE_NUM = 1     # feature (e.g. open high, low, close,volume) number, drop open
FILTERS_NUM = 32   # CNN kernel number
LR = 0.02           # learning rate 0.0002 for CNN, 0.05 for 100 or under, 0.01 for 200 or up 0.0005 for 470 (0.02 opt)
DA = 64           # embedding dimention
D = 16
EPOCH_NUM = 20         # Iteration times for training dat
TICKER_NUM = 50     # S&P500  maximum in data set for 470 tickers for 400 more will be the cudnn bug on batchnorm1d
YEAR_SEED = 0 # train_period = 2010+seed-1-1 to 2010+seed-12-31; test_period = 2011+seed-1-1 to 2011+seed-6-30
DATA_PATH = "/data/kaggle/prices-split-adjusted.csv"
SPY_PATH = '/data/SPY20000101_20171111.csv'
BATCH_SIZE = 128
ATTENTION_LIST = ['BLCNN']#, SELF 'LCNN','BLCNNS','WCC','SWC']#,'LCNN','BLCNNS','WCC','SWC']
LABEL_METHOD = 2 #1: (lable1) same or different sectors; 2: (label2) sector combinations
SAME_SECTOR = True
SIGNIFICANT_A = 0.01
OUT_THRESHOLD = 0.03
FUSIONER = 'DSE' # VOTE or DSE
EVALUATOR_CODES = ['FLUC','ACC_LOSS','ACC_GAIN'] # ACC_LOSS, ACC_GAIN, FLUC
SHOW_ATTENTION = False
REPEAT = 3
PERIOD = ['2010-1-4','2016-12-30']
TEST_RATIO=0.33
# convolution covered WINDOW+n*(WINDOW-1)

class Experiment_platform:

    def __init__(self):
        self.model=None
        self.labels=None
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.datatool = Data_util(TICKER_NUM, WINDOW, FEATURE_NUM, DATA_PATH, SPY_PATH)
        self.cardinal_number,self.train_loader,self.test_loader = self.load_data(PERIOD)
        self.attention_factory=Attention_factory(FEATURE_NUM, FILTERS_NUM, BATCH_SIZE,WINDOW,timestamps=self.train_loader.dataset[0][0].shape[1],sector_cardinality=self.cardinal_number,d=D, da=DA)
        self.abnormal_fusion=Abnormality_fusion(WINDOW)
        self.evaluator_factory = Evaluator_factory()
        self.metrics=Metrics(self.cardinal_number)

    def reset_model(self):
        self.model = None
        self.labels = None
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.attention_factory = Attention_factory(FEATURE_NUM, FILTERS_NUM, BATCH_SIZE, WINDOW,
                                               timestamps=self.train_loader.dataset[0][0].shape[1],
                                               sector_cardinality=self.cardinal_number, d=D, da=DA)

    def reset_abnormal_fusion(self):
        self.abnormal_fusion = Abnormality_fusion(WINDOW)

    def load_data(self,period,batch_size=BATCH_SIZE):
        x = self.datatool.load_x(period)  # load raw data
        x = np.array(x)
        if LABEL_METHOD==1:
            cardinal_number, self.labels = self.Sector_label1()
        elif LABEL_METHOD==2:
            cardinal_number, self.labels = self.Sector_label2()
        else:
            exit()
        print('[Sectors number:',str(cardinal_number),']')
        X_train, X_test, Y_train, Y_test = train_test_split(x, self.labels, test_size=TEST_RATIO,random_state=5)
        train_dataset = PatternDataset(X_train, Y_train)  # labels targets for different loss funtions
        test_dataset = PatternDataset(X_test, Y_test)  # labels targets for different loss funtions
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True)
        return cardinal_number,train_loader,test_loader


    def Sector_label1(self,same_sector=SAME_SECTOR):
        print('[labeling the stocks based on if they have the same sector]')
        labels = []
        for n in range(0, self.datatool.get_dyadic_size()):
            (i, j) = self.datatool.check_dyadic(n)
            ticker1 = self.datatool.check_ticker(i)
            ticker2 = self.datatool.check_ticker(j)
            if self.datatool.check_sector(ticker1) == self.datatool.check_sector(ticker2):
                if same_sector==True:
                     labels.append(1)
                else:
                     labels.append(0)
            else:
                if same_sector==True:
                     labels.append(0)
                else:
                     labels.append(1)
            if n%500==0:
                print('[complete '+str(n)+' of '+str(self.datatool.get_dyadic_size())+' ]')
        labels = np.array(labels)
        return 2,labels

    def Sector_label2(self):
        print('[labeling the stocks based on all their sector combinations]')
        sectors=[]
        labels = []
        for n in range(0, self.datatool.get_dyadic_size()):
            (i, j) = self.datatool.check_dyadic(n)
            ticker1 = self.datatool.check_ticker(i)
            ticker2 = self.datatool.check_ticker(j)
            sector_token=set((self.datatool.check_sector(ticker1),self.datatool.check_sector(ticker2)))

            if sector_token not in sectors:
                sectors.append(sector_token)
            labels.append(sectors.index(sector_token))

            if n%1000==0:
                print('[complete '+str(n)+' of '+str(self.datatool.get_dyadic_size())+' ]')
        labels = np.array(labels)
        return len(sectors),labels


    def abnormality_output(self,train_period,a=0.05):
        spy_data=self.datatool.load_target(train_period)
        evaluators=self.evaluator_factory.get_models(EVALUATOR_CODES)
        #evaluator=self.evaluator_factory.get_model(EVALUATOR_CODE)
        FLUC_score=0
        for evaluator in evaluators:
            evaluator.set_index(spy_data)
            score=evaluator.run(self.abnormal_fusion.Abnormal_period())
            if evaluator.get_code()=='FLUC':
                FLUC_score=score
            print('[%s   %s %f ]' % (
            self.attention_code, evaluator.get_code(), score))
        plt.plot(spy_data,label='SPY')
        plt.xticks([0,250,502,754,1006,1258,1510,1762],['2010','2011','2012','2013','2014','2015','2016','2017'])
        plt.xlabel('Year')
        plt.title("Abnomalities fused by "+FUSIONER+' with '+self.attention_code, fontsize=18)

        plt.legend(loc='upper right')
        sorted_period=sorted(self.abnormal_fusion.Abnormal_period().items(), key=lambda item:item[1], reverse=True)
        n=len(self.abnormal_fusion.Abnormal_period())
        i=0
        for ((window_begin, window_end),value) in sorted_period:
            plt.axvspan(window_begin, window_end, alpha=0.3, color='yellow')
            if i<=int(n*a): i+=1
            else: break
        plt.savefig('result/abnormality_detection_%s_%s_tickernum_%d_FLUC_%f.jpg'%(self.attention_code,FUSIONER,TICKER_NUM,FLUC_score))
        plt.close()



    def train_model(self,lr=LR,epoch=EPOCH_NUM):
        self.model = self.attention_factory.get_model(self.attention_code)
        self.model.cuda()
        if self.attention_code == 'SWC' or self.attention_code == 'WCC':
            return None
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)   # optimize
        #optimizer = torch.optim.Adadelta(self.model.parameters())  # optimize
        # train the deep learning model
        print('[start training the deep learning model]')
        print('[LR=%f epoch=%d]'%(lr,epoch))
        cudnn.benchmark = True

        self.model.train()
        for ep in range(epoch):
            loss_record=[]
            for i, (x, y, indices) in enumerate(self.train_loader):
                if self.attention_code!='CNN' and self.attention_code != 'CNNE':
                    self.model.hidden = self.repackage_hidden(self.model.hidden)
                inputs = Variable(x).float().cuda()
                targets = Variable(y).long().cuda()
                if len(inputs)==BATCH_SIZE:
                    self.model.zero_grad()

                    #attentions,embedding_table,predictions = self.model(inputs)   # model output
                    attentions, predictions = self.model(inputs)  # model output
                    predictions=predictions.squeeze()
                    #convert the prediction into a classification series
                    loss = self.loss_func.forward(predictions, targets)  # loss
                    optimizer.zero_grad()                   # clear gradients for this training step
                    loss.backward()                         # backpropagation, compute gradients retain_graph=True
                    optimizer.step()                        # apply gradients
                    loss_record.append(loss.data.cpu().numpy()[0])
                    if i%100==0:
                        print('step', i,loss.data.cpu().numpy()[0])
            print('*********[epoch %d ave  loss mean %.4f  loss variance %.4f]**************'%(ep,np.array(loss_record).mean(),np.array(loss_record).var()))
        return loss.data.cpu().numpy()[0] # return train loss

    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data).cuda()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def abnormality_detection(self, a):
        self.model.eval()
        self.model.cuda()
        print('[abnormality detection with the learned model]')
        epoch=0
        for i, (x, y, indices) in enumerate(self.train_loader):
            if self.attention_code != 'CNN' and self.attention_code != 'SWC' and self.attention_code !='WCC':
                self.model.hidden = self.repackage_hidden(self.model.hidden)
            if i%100==0:
                print('[Detect the epoch',i,']')
            inputs = Variable(x,volatile=True).float().cuda()
            targets = Variable(y,volatile=True).long().cuda()
            if len(inputs) == BATCH_SIZE:
                attentions, predictions = self.model(inputs)

                if SHOW_ATTENTION is True:
                    self.repeated_display_attention(inputs,attentions,indices,output_number=10,n=5)

                if FUSIONER == 'VOTE':
                    self.abnormal_fusion.vote_statistics(attentions,SIGNIFICANT_A)
                if FUSIONER == 'DSE':
                    self.abnormal_fusion.DS_statistics(attentions,targets,self.cardinal_number,SIGNIFICANT_A)

                epoch+=1
        if FUSIONER == 'DSE':
            self.abnormal_fusion.DS_fusion(self.train_loader.dataset.size())
        self.abnormality_output(self.train_period,a)

    def test_model(self):
        self.model.eval()
        self.model.cuda() # gpu bug
        self.loss_func=self.loss_func.cuda()
        loss_record=[]
        precision_record=[]
        sec_precision_record=[]
        auc_record=[]
        print('[Testing the learned model]')
        for i, (x, y, indices) in enumerate(self.test_loader):
            if self.attention_code != 'CNN' and self.attention_code != 'CNNE':
                self.model.hidden = self.repackage_hidden(self.model.hidden)
            inputs = Variable(x,volatile=True).float().cuda()
            targets = Variable(y,volatile=True).long().cuda()
            if len(inputs) == BATCH_SIZE:
                attentions, predictions = self.model(inputs)
                loss_record.append(self.loss_func.forward(predictions, targets).data.cpu().numpy()[0])
                # cm = confusion_matrix(y_target=targets.data.cpu().numpy(),
                #                       y_predicted=torch.max(predictions, 1)[0].data.cpu().numpy(),
                #                       binary=False)
                precision_record.append(self.metrics.precision(predictions,targets))
                sec_precision_record.append(self.metrics.sector_precision(predictions,targets))
                auc_record.append(self.metrics.sector_auc(predictions,targets))
        print('[Test loss mean %.4f  variance %.4f]'%(np.array(loss_record).mean(),np.array(loss_record).var()))
        print('[Test sector average precision mean %.4f  variance %.4f]' % (
        np.array(sec_precision_record).mean(), np.array(sec_precision_record).var()))
        print('[Test sector AUC mean %.4f  variance %.4f]' % (
            np.array(auc_record).mean(), np.array(auc_record).var()))
        print('[Test precision mean %.4f  variance %.4f]' % (np.array(precision_record).mean(), np.array(precision_record).var()))


    def repeated_display_attention(self,inputs,attentions,indices,output_number,n=5):
        count=output_number
        for i in range(0, len(indices)):
            if count>0:
                self.display_attention(inputs, attentions, i, n)
                count-=1
            else: break


    def display_attention(self,inputs,attentions,i,n=5):
        if i < len(attentions):
            plt.plot(inputs.cpu().data.numpy()[i, 0, :].transpose(),label='time series 1')
            plt.plot(inputs.cpu().data.numpy()[i, 1, :].transpose(), label='time series 2')
            plt.xticks([0,250,502,754,1006,1258,1510,1762],['2010','2011','2012','2013','2014','2015','2016','2017'])
            plt.xlabel('Year')
            plt.title("Dyadic attention based on " + self.attention_code, fontsize=18)
            plt.legend(loc='upper right')
            top_index=attentions[i].sort(descending=True)[1][0:n+1] # find top n value
            attentioned_areas=[]
            for index in top_index:
                index = index.data.cpu().numpy()[0]
                attentioned_areas.append((index,index+WINDOW*3-2))
            for (window_begin,window_end) in attentioned_areas:
                plt.axvspan(window_begin, window_end, alpha=0.5, color='yellow')
            plt.savefig('result/attention__'+self.attention_code+'_tickernum_'+str(TICKER_NUM)+'_dyadic_'+str(i)+'.jpg')
            plt.close()



    def plotweights(self):
        models=[m for m in self.model.modules()]
        cnn1=models[2].cpu()
        cnn2=models[4].cpu()
        weights=cnn1.weight.cumsum(dim=0)[FILTERS_NUM - 1]
        plt.matshow(weights.data.numpy())
        weights=cnn2.weight[0]
        plt.matshow(weights.data.numpy())

    def CPANN_comparison(self,attention_list,repeat=3):
        for i in range(0,repeat):
            print('[Repeat runing %d]' % i)
            for attention_code in attention_list:
                self.reset_model()
                self.attention_code = attention_code
                print('[Run model:', attention_code, ']')
                experiment.reset_abnormal_fusion()
                start = time.clock()
                experiment.train_model()
                elapsed = (time.clock() - start)
                print("[%s] Train used: %.2f Seconds"%(self.attention_code,elapsed))
                start = time.clock()
                experiment.test_model()
                elapsed = (time.clock() - start)
                print("[%s] Test used: %.2f Seconds" %(self.attention_code,elapsed))

                # experiment.test_model()
            torch.cuda.empty_cache()

    def All_comparison(self,attention_list,repeat=3):
        for i in range(0,repeat):
            print('[Repeat runing %d]'%i)
            for attention_code in attention_list:  # 'WCC','BLCNN','LCNN',
                self.reset_model()
                self.attention_code = attention_code
                print('[Run model:', attention_code, ']')
                experiment.reset_abnormal_fusion()
                start = time.clock()
                experiment.train_model()
                elapsed = (time.clock() - start)
                print("Train time used: %.2f Seconds" % elapsed)

                start = time.clock()
                experiment.abnormality_detection(a=OUT_THRESHOLD)
                elapsed = (time.clock() - start)
                print("Detection Time used: %.2f Seconds"%elapsed)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    experiment=Experiment_platform()
    print('[SIGNIFICANT_A %f]'%SIGNIFICANT_A)
    print('[OUT_THRESHOLD %f]'%OUT_THRESHOLD)
    print('[TICKER_NUM %d]'%TICKER_NUM)
    print('[FUSIONER: %s]'%FUSIONER)
    print('[LABEL METHOD: %d]' % LABEL_METHOD)
    print('[TRAIN, TEST PERIOD: %s]' % PERIOD)
    print('[TEST_RATIO: %f]' % TEST_RATIO)
    #experiment.All_comparison(ATTENTION_LIST,repeat=REPEAT)
    experiment.CPANN_comparison(ATTENTION_LIST,repeat=REPEAT)

    #experiment.plotweights()
