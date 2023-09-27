from copy import deepcopy
import pickle
import os

from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm

from settings import RANDOM_STATE


def reg_stacking_fit(base_models, meta_model, model_names, X, y, folder_path, nfolds=5):
    meta_X = np.zeros((len(X), len(base_models)))
    meta_y = np.zeros(len(X))
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=RANDOM_STATE)
    for i, (train_index, test_index) in tqdm(enumerate(kf.split(X))):
        train_X, train_y = X[train_index], y[train_index]
        test_X, test_y = X[test_index], y[test_index]

        for idx, model in enumerate(base_models):
            fold_model = deepcopy(model)
            fold_model.fit(train_X, train_y)
            preds = fold_model.predict(test_X)
            meta_X[test_index, idx] = preds
            with open(
                os.path.join(folder_path, f"{model_names[idx]}_{i}.pickle"), "wb"
            ) as fin:
                pickle.dump(fold_model, fin)

        meta_y[test_index] = test_y
    meta_model.fit(meta_X, meta_y)
    with open(os.path.join(folder_path, f"meta_{model_names[-1]}.pickle"), "wb") as fin:
        pickle.dump(fold_model, fin)


def reg_stacking_predict(model_names, X, folder_path, nfolds=5):
    meta_X = np.zeros((len(X), len(model_names) - 1))
    for idx, _ in enumerate(model_names[:-1]):
        preds = np.zeros(len(X))
        for i in range(nfolds):
            with open(
                os.path.join(folder_path, f"{model_names[idx]}_{i}.pickle"), "rb"
            ) as fout:
                fold_model = pickle.load(fout)
            preds += fold_model.predict(X)
        preds /= nfolds
        meta_X[:, idx] = preds
    with open(
        os.path.join(folder_path, f"meta_{model_names[-1]}.pickle"), "rb"
    ) as fout:
        meta_model = pickle.load(fout)
    meta_preds = meta_model.predict(meta_X)
    return meta_preds


def clf_stacking_fit(base_models, meta_model, model_names, X, y, folder_path, nfolds=5):
    meta_X = np.zeros((len(X), len(base_models)))
    meta_y = np.zeros(len(X))
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=RANDOM_STATE)
    for i, (train_index, test_index) in tqdm(enumerate(kf.split(X))):
        train_X, train_y = X[train_index], y[train_index]
        test_X, test_y = X[test_index], y[test_index]

        for idx, model in enumerate(base_models):
            fold_model = deepcopy(model)
            fold_model.fit(train_X, train_y)
            preds = fold_model.predict_proba(test_X)[:, 1]
            meta_X[test_index, idx] = preds
            with open(
                os.path.join(folder_path, f"{model_names[idx]}_{i}.pickle"), "wb"
            ) as fin:
                pickle.dump(fold_model, fin)

        meta_y[test_index] = test_y
    meta_model.fit(meta_X, meta_y)
    with open(os.path.join(folder_path, f"meta_{model_names[-1]}.pickle"), "wb") as fin:
        pickle.dump(fold_model, fin)


def clf_stacking_predict(model_names, X, folder_path, nfolds=5):
    meta_X = np.zeros((len(X), len(model_names) - 1))
    for idx, _ in enumerate(model_names[:-1]):
        preds = np.zeros(len(X))
        for i in range(nfolds):
            with open(
                os.path.join(folder_path, f"{model_names[idx]}_{i}.pickle"), "rb"
            ) as fout:
                fold_model = pickle.load(fout)
            preds += fold_model.predict_proba(X)[:, 1]
        preds /= nfolds
        meta_X[:, idx] = preds
    with open(
        os.path.join(folder_path, f"meta_{model_names[-1]}.pickle"), "rb"
    ) as fout:
        meta_model = pickle.load(fout)
    meta_preds = meta_model.predict(meta_X)
    return meta_preds
