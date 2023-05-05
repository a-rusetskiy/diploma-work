from sklearn.metrics import mean_absolute_error as mae

from utils import utils
from utils.deep_learning import LightningNet, GridSearch

KFOLD_VALUE = 2
MODEL_NAME = 'NN'  # 'LSTM'
VERBOSE = True


if __name__ == '__main__':
    data = utils.get_data()
    model_parameters = utils.get_model_parameters(MODEL_NAME)

    clf = GridSearch(model_type=MODEL_NAME, model=LightningNet, model_parameters=model_parameters,
                     cv=KFOLD_VALUE, loss=mae, verbose=VERBOSE)
    clf.fit(data)

    utils.save_best_parameters(MODEL_NAME, best_parameters=clf.best_parameters)
    utils.save_model(model=clf.best_model, model_name=MODEL_NAME)
