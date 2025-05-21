import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    if len(feature_vector) != len(target_vector):
        print('bad dim') # если не совпадают размерности, то выводим ошибку
        return

    ftrs_sorted_index = np.argsort(feature_vector)
    feature_vector_s = feature_vector[ftrs_sorted_index]# сортируем нашу фичу и берем индексы, это нужно, чтобы потом смотреть на порог разбиения
    target_vector_s = target_vector[ftrs_sorted_index]
    
    norm_idx= np.where(feature_vector_s[:-1] != feature_vector_s[1:])[0] # берем те индексы, где у нас различные значения и все упорядочено
    if len(norm_idx) == 0:
        return np.array([]), np.array([]), None, np.inf
    
    thresholds = (feature_vector_s[norm_idx] + feature_vector_s[norm_idx + 1]) / 2 # берем средние значения между вумя соседними значениями отсортированного признака
    gini = []# делаем массив метрики
    n = len(target_vector_s)
    
    for i in thresholds: #перебираем пороги
        l = feature_vector_s < i
        r = feature_vector_s >= i # делим на левые значения и правые
        if l.sum() == 0 or r.sum() == 0:
            gini.append(np.inf) # если хоть одно из разбиений только 0, то это бесконечность
            continue
        l_p0 = np.mean(target_vector_s[l])
        l_p1 = 1 - l_p0
        h_l = 1 - (l_p0**2) - (l_p1**2) # берем энтропию разбиения слева
        
        r_p0 = np.mean(target_vector_s[r])
        r_p1 = 1 - r_p0
        h_r = 1 - (r_p0**2) - (r_p1**2) # энтропия разбиения справа
        
        g = (l.sum() / n) * h_l + (r.sum() / n) * h_r # считаем джини
        gini.append(g)
    
    gini = np.array(gini)# сортируем, чтобы потом взять лучшие значения
    best_idx = np.argmin(gini)
    
    return thresholds, gini, thresholds[best_idx], gini[best_idx]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("Неизвестный тип признака")

        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._tree = {}

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = int(sub_y[0])
            return

        if (self._max_depth is not None and depth >= self._max_depth) or len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        categories_map_best = None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)
            else:  # categorical
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {k: clicks.get(k, 0) / counts[k] for k in counts}
                sorted_cats = sorted(ratio, key=ratio.get)
                categories_map = {cat: i for i, cat in enumerate(sorted_cats)}
                feature_vector = np.array([categories_map.get(v, -1) for v in sub_X[:, feature]])

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if threshold is None or gini is None:
                continue

            if gini_best is None or gini < gini_best:
                feature_best = feature
                gini_best = gini
                threshold_best = threshold

                if feature_type == "real":
                    split = feature_vector < threshold
                    categories_map_best = None
                else:
                    split = feature_vector < threshold
                    categories_map_best = categories_map

        if feature_best is None or split.sum() < self._min_samples_leaf or (~split).sum() < self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        feature_type = self._feature_types[feature_best]

        if feature_type == "real":
            node["threshold"] = float(threshold_best)
        else:
            node["categories_split"] = [k for k, v in categories_map_best.items() if v < threshold_best]

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]

        if feature_type == "real":
            threshold = node.get("threshold")
            return self._predict_node(
                x, node["left_child"] if float(x[feature_idx]) < threshold else node["right_child"]
            )
        else:
            category = x[feature_idx]
            left_categories = node.get("categories_split", [])
            return self._predict_node(
                x, node["left_child"] if category in left_categories else node["right_child"]
            )

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X], dtype=int)
