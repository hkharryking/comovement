class Evaluator:
    def __init__(self):
        pass

    def set_index(self,data):
        self.index_data=data

    def run(self,periods):
        return None

    def period_combine(self,periods):
        combined_periods=[]
        initial=True
        for (start,end) in periods:
            if initial is True:
                start_n=start
                end_n=end
                initial=False
                continue
            if start<=start_n and end >=start_n:
                start_n=start
            if start<=end_n and end>=end_n:
                end_n=end
            if end<start_n or start>end_n:
                combined_periods.append((start_n,end_n))
                start_n=start
                end_n=end
        combined_periods.append((start_n,end_n))
        return combined_periods

if __name__ == "__main__":
    ee=Evaluator()
    l=[(1,4),(2,5),(7,12),(10,23)]
    ee.period_combine(l)


class Fluctuation_evaluator(Evaluator):
    def get_code(self):
        return 'FLUC'

    def run(self,periods):
        periods=self.period_combine(periods)
        result=0
        for (start,end) in periods:
            loss=self.index_data[start:end].max()-self.index_data[start:end].min()
            #loss*=periods[(start,end)]
            result+=loss
        return result

class Accumulative_loss_evaluator(Evaluator):
    def get_code(self):
        return 'ACC_LOSS'

    def run(self,periods):
        periods = self.period_combine(periods)
        result=0
        for (start,end) in periods:
            window=self.index_data[start:end]
            for i in range(0,len(window)):
                max_window_loss=0
                for j in range(i,len(window)):
                    window_loss = window[i]-window[j]
                    if window[i]>window[j] and max_window_loss<window_loss:
                        max_window_loss=window_loss

                result+=max_window_loss
        return result

class Accumulative_gain_evaluator(Evaluator):
    def get_code(self):
        return 'ACC_GAIN'

    def run(self,periods):
        periods = self.period_combine(periods)
        result=0
        for (start,end) in periods:
            window=self.index_data[start:end]
            for i in range(0,len(window)):
                max_window_gain = 0
                for j in range(i,len(window)):
                    window_gain = window[j] - window[i]
                    if window[i]<window[j] and max_window_gain<window_gain:
                        max_window_gain=window_gain

                result+=max_window_gain
        return result
