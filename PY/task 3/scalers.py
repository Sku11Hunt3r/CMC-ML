import numpy as np
import typing


class MinMaxScaler:
    def fit(self, data: np.ndarray) -> None:
        self.min = np.zeros(len(data[0]))
        self.max = np.zeros(len(data[0]))
        for j in range(len(data[0])):
            self.min[j] = data[0][j]
            self.max[j] = data[0][j]
            for i in range(len(data)):
                if (data[i][j] < self.min[j]):
                    self.min[j] = data[i][j]
                if (data[i][j] > self.max[j]):
                    self.max[j] = data[i][j]

    def transform(self, data: np.ndarray) -> np.ndarray:
        for j in range(len(data[0])):
            for i in range(len(data)):
                data[i][j] = (data[i][j] - self.min[j])/(self.max[j] - self.min[j])
        return data


class StandardScaler:
    def fit(self, data: np.ndarray) -> None:
        self.E = np.zeros(len(data[0]))
        self.D = np.zeros(len(data[0]))
        for j in range(len(data[0])):
            exp = 0
            disp = 0
            for i in range(len(data)):
                exp += data[i][j]
            exp /= len(data)
            for i in range(len(data)):
                disp += (data[i][j] - exp)**2
            disp /= len(data)
            disp = disp ** (1/2)
            self.E[j] = exp
            self.D[j] = disp

    def transform(self, data: np.ndarray) -> np.ndarray:
        for j in range(len(data[0])):
            for i in range(len(data)):
                data[i][j] = (data[i][j] - self.E[j]) / self.D[j]
        return data
