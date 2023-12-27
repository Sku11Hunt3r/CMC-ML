import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    fold_len = num_objects // num_folds
    indexValues = np.arange(num_objects)
    res = []
    for i in range(num_folds - 1):
        test = indexValues[i*fold_len: (i+1) * fold_len]
        turp = (np.delete(indexValues, test), test)
        res.append(turp)
    last_test = indexValues[(num_folds - 1) * fold_len:]
    res.append((np.delete(indexValues, last_test), last_test))
    return res


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    res = {}
    fold_res = np.zeros(len(folds))
    for neig in parameters['n_neighbors']:
        for m in parameters['metrics']:
            for w in parameters['weights']:
                for norm in parameters['normalizers']:
                    for i in range(len(folds)):
                        model = knn_class(n_neighbors=neig, metric=m, weights=w)
                        if norm[0] is not None:
                            norm[0].fit(X[folds[i][0]])
                            new_X = norm[0].transform(X[folds[i][0]])
                            model.fit(new_X, y[folds[i][0]])
                            prediction = model.predict(norm[0].transform(X[folds[i][1]]))
                            fold_res[i] = score_function(y[folds[i][1]], prediction)
                        else:
                            model.fit(X[folds[i][0]], y[folds[i][0]])
                            prediction = model.predict(X[folds[i][1]])
                            fold_res[i] = score_function(y[folds[i][1]], prediction)
                    res[(norm[1], neig, m, w)] = fold_res.mean()
    return res
