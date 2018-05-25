from crnn_stock.attention_cnne import Attention_CNNE
from crnn_stock.attention_lcnn import Attention_LCNN
from crnn_stock.attention_blcnn import Attention_BLCNN
from crnn_stock.attention_self import Attention_SELF
from crnn_stock.attention_wlcnn import Attention_WLCNN
from crnn_stock.attention_swc import Attention_SWC
from crnn_stock.attention_wcc import Attention_WCC
from crnn_stock.attention_cnn import Attention_CNN

class Attention_factory:
    def __init__(self,feature_size, filer_num, batch_size, window, timestamps,sector_cardinality, d, da):
        self.attentions = []
        self.attentions.append((Attention_CNNE(feature_size, filer_num, batch_size, window, timestamps,sector_cardinality, d, da)))
        self.attentions.append(
            (Attention_CNN(feature_size, filer_num, batch_size, window, timestamps, sector_cardinality, d, da)))
        self.attentions.append((Attention_LCNN(feature_size, filer_num, batch_size, window, timestamps,sector_cardinality, d, da)))
        self.attentions.append((Attention_BLCNN(feature_size, filer_num, batch_size, window, timestamps,sector_cardinality, d, da)))
        self.attentions.append(
            (Attention_SELF(feature_size, filer_num, batch_size, window, timestamps, sector_cardinality, d, da)))
        self.attentions.append(
            (Attention_WLCNN(feature_size, filer_num, batch_size, window, timestamps, sector_cardinality, d, da)))
        self.attentions.append((Attention_SWC(window)))
        self.attentions.append((Attention_WCC(window)))

    def get_model(self,code):
        for model in self.attentions:
            if model.get_code()==code:
                return model
        return None