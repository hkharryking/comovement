from crnn_stock.evaluator import Accumulative_gain_evaluator,Accumulative_loss_evaluator,Fluctuation_evaluator

class Evaluator_factory:
    def __init__(self):
        self.evaluators = []
        self.evaluators.append(Fluctuation_evaluator())
        self.evaluators.append(Accumulative_loss_evaluator())
        self.evaluators.append(Accumulative_gain_evaluator())

    def get_model(self,code):
        for model in self.evaluators:
            if model.get_code()==code:
                return model
        return None

    def get_models(self, model_codes):
        models = []
        for code in model_codes:
            models.append(self.get_model(code))
        return models