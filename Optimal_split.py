import numpy as np


class Optimal_split:

    def split_node(self, R, y, feature, t):
        right_indices = R[feature] > t
        left_indices = R[feature] < t
        R_l, y_l = R[left_indices], y[left_indices]
        R_r, y_r = R[right_indices], y[right_indices]
        return R_l, R_r, y_l, y_r

    def H(self, R, y):
        y_R = y[R.index]
        unique_elem, quantity = np.unique(y_R, return_counts=True)
        p = quantity / y_R.shape[0]
        return np.sum(p * (1 - p))

    def q_error(self, R, y, feature, t):
        R_l, R_r, y_l, y_r = self.split_node(R, y, feature, t)
        Q = (len(R_l) / len(R)) * self.H(R_l, y) + (len(R_r) / len(R)) * self.H(R_r, y)
        return Q

    def get_optimal_split_one_feature(self, R, y, feature):
        Q_array = []
        feature_values = np.unique(R[feature])
        for t in feature_values:
            Q_array.append(self.q_error(R, y, feature, t))
        opt_threshold = feature_values[np.argmax(Q_array)]
        return opt_threshold

    def get_optimal_split(self, R, y):
        min_t = 9999
        Feature = ''
        for feature in R.columns:
            t = self.get_optimal_split_one_feature(R, y, feature)
            if t < min_t:
                min_t = t
                Feature = feature
        return (Feature, min_t)

    def leaf_value(self, y):
        y = np.asarray(y, dtype=int)
        classes, leaf_counts = np.unique(y, return_counts=True)
        leaf_counts = np.array(leaf_counts)
        most_common = classes[np.argmax(leaf_counts)]
        return most_common

    def leaf(self, x, y):

        feat_t = self.get_optimal_split(x, y)
        R_l, R_r, y_l, y_r = self.split_node(x, y, *feat_t)
        return (R_l, y_l), (R_r, y_r)

