###########imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn import preprocessing as pp
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


###########func
def kFolds(data, fold):
    c = int(data.shape[0] / fold)
    kf = KFold(n_splits=fold)
    m = np.empty(0)
    for train, test in kf.split(data):
        m = np.append(m, test)
    if data.shape[0] % 10 > 0:
        for i in range(data.shape[0] % 10):
            m = np.delete(m, m.size - 1)
    m = m.reshape((c, fold), order='F')
    return m


def testTrainChop1(data, fold, component, lableColumnNum):
    acc = 0
    for i in range(fold.shape[1]):
        xtrain = np.delete(fold, i, axis=1).flatten()
        xtest = fold[:, i]
        Xtrain = pd.DataFrame(data[np.array(xtrain, dtype=int)])
        x_group_train = Xtrain.groupby(lableColumnNum)
        x_train_one = x_group_train.get_group(1)
        x_train_sec = x_group_train.get_group(2)
        x_train_one_data = x_train_one.values[:, :lableColumnNum]
        x_train_sec_data = x_train_sec.values[:, :lableColumnNum]
        Xtest = data[np.array(xtest, dtype=int)]
        gmm1 = GaussianMixture(n_components=component, covariance_type='diag')
        gmm1.fit(x_train_one_data)
        gmm2 = GaussianMixture(n_components=component, covariance_type='diag')
        gmm2.fit(x_train_sec_data)
        for item in Xtest:
            tmp = np.array(xtest[:lableColumnNum]).reshape((-1, 1))
            prd1 = gmm1.predict(tmp.T)
            prd2 = gmm2.predict(tmp.T)
            if np.argmax(np.array([0, prd1, prd2])) == item[lableColumnNum]:
                acc += 1
    return (acc * 100) / data.shape[0]


def testTrainChop2(data, fold, component, lableColumnNum):
    acc = 0
    for i in range(fold.shape[1]):
        xtrain = np.delete(fold, i, axis=1).flatten()
        xtest = fold[:, i]
        Xtrain = pd.DataFrame(data[np.array(xtrain, dtype=int)])
        x_group_train = Xtrain.groupby(lableColumnNum)
        x_train_one = x_group_train.get_group(1)
        x_train_sec = x_group_train.get_group(2)
        x_train_third = x_group_train.get_group(3)
        x_train_fourth = x_group_train.get_group(4)
        x_train_one_data = x_train_one.values[:, :lableColumnNum]
        x_train_sec_data = x_train_sec.values[:, :lableColumnNum]
        x_train_third_data = x_train_third.values[:, :lableColumnNum]
        x_train_fourth_data = x_train_fourth.values[:, :lableColumnNum]
        Xtest = data[np.array(xtest, dtype=int)]
        gmm1 = GaussianMixture(n_components=component, covariance_type='diag')
        gmm1.fit(x_train_one_data)
        gmm2 = GaussianMixture(n_components=component, covariance_type='diag')
        gmm2.fit(x_train_sec_data)
        gmm3 = GaussianMixture(n_components=component, covariance_type='diag')
        gmm3.fit(x_train_third_data)
        gmm4 = GaussianMixture(n_components=component, covariance_type='diag')
        gmm4.fit(x_train_fourth_data)

        for item in Xtest:
            tmp = np.array(xtest[:lableColumnNum]).reshape((-1, 1))
            prd1 = gmm1.predict(tmp.T)
            prd2 = gmm2.predict(tmp.T)
            prd3 = gmm3.predict(tmp.T)
            prd4 = gmm4.predict(tmp.T)
            if np.argmax(np.array([0, prd1, prd2, prd3, prd4])) == item[lableColumnNum]:
                acc += 1
    return (acc * 100) / data.shape[0]


data_heart = pd.read_csv('datasets/heart.dat', header=None, sep=' ')
data_heart_values = data_heart.values[:, :13]
data_heart_label = np.array(data_heart.values[:, -1])

data_heart_label = data_heart_label.reshape((-1, 1))
data_heart_normalize = pp.scale(data_heart_values, axis=1)
data_heart_normalize = np.concatenate((data_heart_normalize, data_heart_label), axis=1)
heartFold = kFolds(data_heart_normalize, 10)
print("data heart accuracy k=1:", testTrainChop1(data_heart_normalize, heartFold, 1, 13))
print("data heart accuracy k=2:", testTrainChop1(data_heart_normalize, heartFold, 2, 13))
print("data heart accuracy:k=4", testTrainChop1(data_heart_normalize, heartFold, 4, 13))
print("data heart accuracy:k=8", testTrainChop1(data_heart_normalize, heartFold, 8, 13))
print("data heart accuracy:k=12", testTrainChop1(data_heart_normalize, heartFold, 12, 13))
print("data heart accuracy:k=20", testTrainChop1(data_heart_normalize, heartFold, 20, 13))
data_vehicle = pd.read_csv('datasets/xaa.dat', header=None, sep=' ')
data_vehicle.dropna(inplace=True, axis=1)
data_vehicle_values = data_vehicle.values[:, :18]
data_vehicle_label = np.array(data_vehicle.values[:, -1])
data_vehicle_label = data_vehicle_label.reshape((-1, 1))
for datal in range(data_vehicle_label.shape[0]):
    if data_vehicle_label[datal] == 'van':
        data_vehicle_label[datal] = 1
    if data_vehicle_label[datal] == 'bus':
        data_vehicle_label[datal] = 2
    if data_vehicle_label[datal] == 'opel':
        data_vehicle_label[datal] = 3
    if data_vehicle_label[datal] == 'saab':
        data_vehicle_label[datal] = 4

data_vehicle_normalize = pp.scale(data_vehicle_values, axis=1)
data_vehicle_normalize = np.concatenate((data_vehicle_normalize, data_vehicle_label), axis=1)

vehicleFold = kFolds(data_vehicle_normalize, 10)
print("data vehicle accuracy", testTrainChop2(data_vehicle_normalize, vehicleFold, 10, 18))
