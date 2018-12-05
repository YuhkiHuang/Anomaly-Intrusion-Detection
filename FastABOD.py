class FastABOD:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit_predict(self, X, contamination=0.1):
        # 找出各点的k个最近邻居点
        k_nearest = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        distances, indices = k_nearest.kneighbors(X)
        # 计算各点与其他数据对的角度方差
        numbers = [i + 1 for i in range(distances.shape[1] - 1)]
        combs = list(itertools.combinations(numbers, 2))
        # 定义ABOF
        abofs = []
        for i in range(len(X)):
            x = X[indices[i]]
            abof = self._compute_abof(x, combs)
            abofs.append(abof)
        # ABOF中定义序列中得分低的N%个数据为异常数据
        ordered_abofs = np.argsort(abofs)
        anomaly_indices = ordered_abofs[:int(len(abofs)*contamination + 0.5)]
        # scikit-learn中正常数据返回1，异常数据返回
        prediction = np.ones((len(abofs)), dtype=np.int)
        prediction[anomaly_indices] = -1
        return prediction

    def _compute_abof(self, x, combs):
        numerator1 = 0
        numerator2 = 0
        denominator1 = 0
        for comb in combs:
            AB = x[comb[0]] - x[0]
            AC = x[comb[1]] - x[0]
            AB_norm = np.linalg.norm(AB)
            AC_norm = np.linalg.norm(AC)
            if AB_norm == 0 or AC_norm == 0:
                #pass
                continue
            a = 1 / (AB_norm * AC_norm)
            b = np.dot(AB, AC) / ((AB_norm ** 2) * (AC_norm ** 2))
            numerator1 += a * (b ** 2)
            denominator1 += a
            numerator2 += a * b
        denominator2 = denominator1
        return numerator1 / denominator1 - (numerator2 / denominator2) ** 2
