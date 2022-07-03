def find_best_split(feature_vector, target_vector):
    th0, _ = np.unique(feature_vector, return_counts=True)
    th1 = np.append(0, th0)[:-1]
    thresholds = ((th0+th1)*1./2)[1:]
    tmp_mass = np.append(feature_vector, th0)
    tmp_target = np.append(target_vector, np.ones(len(th0)))
    _, var_cumsum = np.unique(tmp_mass[np.where(tmp_target == 1)], return_counts=True)
    a_unique, value_count = np.unique(feature_vector, return_counts=True)

    def H(p_1):
        return (1-p_1**2-(1-p_1)**2)

    H_l_1 = (np.cumsum(var_cumsum-1)*1./np.cumsum(value_count))[:-1]
    H_r_1 = (np.cumsum(var_cumsum[::-1]-1) * 1. / np.cumsum(value_count[::-1]))[:-1][::-1]
    left_delitel = (np.cumsum(value_count) * 1. / len(target_vector))[:-1]
    ginis = -left_delitel * H(H_l_1) - (1-left_delitel) * H(H_r_1)
    if len(np.unique(feature_vector)) == 1:
        return [-1]*len(feature_vector), [-1]*len(feature_vector), -1, -1
    else:
        return thresholds, ginis, thresholds[np.argmax(ginis)], np.max(ginis)


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types, max_depth=None,
                 min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical",
                           feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self.feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click/current_count
                sorted_categories = list(map(lambda x: x[0],
                                             sorted(ratio.items(),
                                                    key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories,
                                          list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(feature_vector) == 0:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, np.array(sub_y))
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold,
                                                     categories_map.items())))
                else:
                    raise ValueError

        if gini_best == -1:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)],
                       sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        x = np.array(x)
        tree = node
        class_i = None
        for i in range(100):
            if tree['type'] == 'terminal':
                class_i = tree['class']
                break
            else:
                feature_split_i = tree['feature_split']
                if self.feature_types[tree['feature_split']] == 'real':
                    real_split_i = tree['threshold']
                    if x[feature_split_i] < real_split_i:
                        tree = tree['left_child']
                    else:
                        tree = tree['right_child']
                else:
                    categories_split_i = tree['categories_split']
                    if x[feature_split_i] in categories_split_i:
                        tree = tree['left_child']
                    else:
                        tree = tree['right_child']
        return class_i

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)