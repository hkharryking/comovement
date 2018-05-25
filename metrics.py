from sklearn import metrics


class Metrics:
    def __init__(self,cardinal_number):
        self.cardinal_number=cardinal_number

    '''
    Total number of the correct predicted sectors
    '''
    def precision(self,predictions,targets):
        correct = targets.eq(predictions.max(dim=1)[1]).sum()
        acc = (correct.float() / len(targets)).cpu().data.numpy()
        return acc

    '''
    Average correct number for each sector
    '''
    def sector_precision(self,predictions,targets):
        acc=0.
        sector_number=0
        for s in range(0,self.cardinal_number):
            indices = (targets==s)
            total = indices.nonzero().data.cpu().numpy().shape[0]
            if total > 0:
                correct = (predictions.max(dim=1)[1][indices]==s).nonzero().sum()
                acc+=int(correct.cpu().data.numpy())/total
                sector_number+=1
        return acc/sector_number

    '''
    Average AUC for each sector
    '''
    def sector_auc(self,predictions,targets):
        y=targets.data.cpu().numpy()
        pred=predictions.data.cpu().numpy()
        auc=0.
        sector_number = 0
        for s in range(0, self.cardinal_number):
            if int((targets==s).nonzero().sum().data.cpu().numpy())>0:
                fpr, tpr, thresholds = metrics.roc_curve(y, pred.transpose()[s], pos_label=s)
                auc+=metrics.auc(fpr, tpr)
                sector_number+=1
        return auc / sector_number

    def auc(self,prediction,target):
        fpr, tpr, thresholds = metrics.roc_curve(target, prediction, pos_label=1)
        auc=metrics.auc(fpr, tpr)
        return auc