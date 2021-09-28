# Carlos Hernani Morales, carhermo@alumni.uv.es 2021

import numpy as np
import pandas as pd
from pandas.io.stata import precision_loss_doc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.svm import SVR

def load_data(data_path, is_numpy=True):
    if is_numpy:
        with np.load(data_path, allow_pickle=True) as data:
            data = data[data.files[0]]
    else:
        pass
    return data

def get_dataframe(data, param_names):
    """
    Our data is usually like this:
    numpy.array (1000, n)
    where n = parameters + xdata
    parameters store several values
    """
    number_columns = list(range(data.shape[1]))
    param_columns = number_columns[:-1]

    try:
        parameters = [
            pd.DataFrame(
                [row.flatten() for row in data[:,par]]
                ) for par in param_columns
            ]
    except:
        parameters = [
            pd.DataFrame(
                [row for row in data[:,par]]
                ) for par in param_columns
            ]
    
    x = pd.DataFrame([row.flatten() for row in data[:,-1]])
    
    params = pd.concat(parameters, axis=1)
    params.columns = param_names
    return x, params

def pca(x, min_evr=0.999):
    p = PCA(n_components=min_evr)
    return p.fit_transform(x)

def regressor(estimator, xtrain, xtest, ytrain, ytest, oaa=True):
    model = []
    score = []
    preds = []
    if oaa: # regression of one variable each time
        for col in ytrain.columns:
            model.append(estimator.fit(xtrain, ytrain[col]))
            score.append(estimator.score(xtest, ytest[col]))
            preds.append(pd.DataFrame(estimator.predict(xtest), columns=[col]))
    else:
        model.append(estimator.fit(xtrain, ytrain))
        score.append(estimator.score(xtest, ytest))
        preds.append(pd.DataFrame(estimator.predict(xtest),columns=ytest.columns))
    preds = pd.concat(preds, axis=1)
    return {'model': model, 'score': score, 'preds': preds, 'real': pd.DataFrame(ytest,columns=ytest.columns)}

def etr_reg(xy, col=None, n_estimators=400, random_state=0, oaa=False):
    return regressor(etr(n_estimators=n_estimators, random_state=random_state, bootstrap=False), *xy, oaa=oaa)

def svr_reg(xy, **kwargs):
    return regressor(SVR(**kwargs), *xy)


def norm(predictions):
    new_r = predictions.copy(deep=True)
    new_r['r.n'] = new_r.filter(regex=("r\.*")).apply(lambda x: x**2).apply(lambda x: np.sqrt(np.sum(x)), axis=1)
    new_r = new_r.filter(regex=("r\.*")).apply(lambda x: x/x['r.n'], axis=1)
    #new_r.columns = ['n'+ col for col in predictions.filter(regex=("r\.*")).columns]
    return pd.concat([predictions.filter(regex=("a\.*")), new_r], axis=1).drop(['r.n'], axis=1)
    


def fidelity_1(real, predictions):
    """
    Fidelity calculation for 4 Modes 1 photon
    """
    real_a0 = real['a.0'].to_numpy(dtype = 'complex128')
    real_a1 = real['a.1'].to_numpy(dtype = 'complex128')
    pred_a0 = predictions['a.0'].to_numpy(dtype = 'complex128')
    pred_a1 = predictions['a.1'].to_numpy(dtype = 'complex128')
    p0 = np.cos(real_a0)*np.cos(pred_a0)
    p1 = np.sin(real_a0)*np.sin(pred_a0)
    #p0 *= 1
    p1 *= np.exp(1j*(real_a1 - pred_a1))
    return np.abs(p0+p1)

def fidelity_2(real, predictions):
    """
    Fidelity calculation for 4 Modes 2 photons
    """
    real_r = real.filter(regex=("r\.*")).reset_index().drop(['index'], axis=1)
    real_a = real.filter(regex=("a\.*")).reset_index().drop(['index'], axis=1)
    pred_r = predictions.filter(regex=("r\.*")).reset_index().drop(['index'], axis=1)
    pred_a = predictions.filter(regex=("a\.*")).reset_index().drop(['index'], axis=1)

    rr = real_r.mul(pred_r)
    aa = real_a - pred_a
    rr = rr.to_numpy(dtype = 'complex128')
    aa = aa.to_numpy(dtype = 'complex128')
    eaa = np.exp(1j*aa)
    rr_eaa = np.multiply(rr, eaa)
    return np.abs(np.sum(rr_eaa,axis=1))

def scores(models):
    if not isinstance(models, (list, tuple)):
        models = [models]
    _ = [print('Model: {} \t Score: {}\n'.format(mod['model'], mod['score'])) for mod in models]



