import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from utils import utils

np.random.seed(123)
ERROR_NAME = 'neg_mean_absolute_error'
MODEL_NAME = 'random_forest'
KFOLD_VALUE = 10
VERBOSE = 10


if __name__ == '__main__':
    df = utils.get_data()
    X, y = utils.target_split(df)

    model_parameters = utils.get_model_parameters(MODEL_NAME)
    model = RandomForestRegressor()
    clf = GridSearchCV(model, param_grid=model_parameters, cv=KFOLD_VALUE,
                       scoring=ERROR_NAME, verbose=VERBOSE, n_jobs=-1)
    clf.fit(X, y)

    utils.save_best_parameters(MODEL_NAME, best_parameters=clf.best_params_)
    utils.save_model(model=clf.best_estimator_, model_name=MODEL_NAME)
