import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression
import math


def loadData():
    labels = [0, 1, 2, 3, 4, 5, 6, 7]
    path = r"H:\uni\Alzahra\8\Python\HW\04\Tennis-Major-Tournaments-Match-Statistics\AusOpen-men-2013.csv"
    data = pd.read_csv(path)
    data['Label'] = labels[0]
    data.fillna(data.mean(), inplace=True)

    path = r"H:\uni\Alzahra\8\Python\HW\04\Tennis-Major-Tournaments-Match-Statistics\AusOpen-women-2013.csv"
    alternativeData = pd.read_csv(path)
    alternativeData['Label'] = labels[1]
    alternativeData.fillna(data.mean(), inplace=True)

    data12 = data.append(alternativeData, ignore_index=True)

    path = r"H:\uni\Alzahra\8\Python\HW\04\Tennis-Major-Tournaments-Match-Statistics\FrenchOpen-men-2013.csv"
    data = pd.read_csv(path)
    data['Label'] = labels[2]
    data.fillna(data.mean(), inplace=True)

    path = r"H:\uni\Alzahra\8\Python\HW\04\Tennis-Major-Tournaments-Match-Statistics\FrenchOpen-women-2013.csv"
    alternativeData = pd.read_csv(path)
    alternativeData['Label'] = labels[3]
    alternativeData.fillna(data.mean(), inplace=True)

    data14 = data12.append(data.append(alternativeData, ignore_index=True), ignore_index=True)

    path = r"H:\uni\Alzahra\8\Python\HW\04\Tennis-Major-Tournaments-Match-Statistics\USOpen-men-2013.csv"
    data = pd.read_csv(path)
    data['Label'] = labels[4]
    data.fillna(data.mean(), inplace=True)

    path = r"H:\uni\Alzahra\8\Python\HW\04\Tennis-Major-Tournaments-Match-Statistics\USOpen-women-2013.csv"
    alternativeData = pd.read_csv(path)
    alternativeData['Label'] = labels[5]
    alternativeData.fillna(data.mean(), inplace=True)

    data16 = data14.append(data.append(alternativeData, ignore_index=True), ignore_index=True)

    path = r"H:\uni\Alzahra\8\Python\HW\04\Tennis-Major-Tournaments-Match-Statistics\Wimbledon-men-2013.csv"
    data = pd.read_csv(path)
    data['Label'] = labels[6]
    data.fillna(data.mean(), inplace=True)

    path = r"H:\uni\Alzahra\8\Python\HW\04\Tennis-Major-Tournaments-Match-Statistics\Wimbledon-women-2013.csv"
    alternativeData = pd.read_csv(path)
    alternativeData['Label'] = labels[7]
    alternativeData.fillna(data.mean(), inplace=True)

    finalData = data16.append(data.append(alternativeData, ignore_index=True), ignore_index=True)

    finalData.drop('ST2.1.1', inplace=True, axis=1)
    finalData.drop('ST1.1.1', inplace=True, axis=1)
    print(finalData)

    return finalData


def analyzeData(dataset):
    print("Dataset dimensions is: ")
    print(dataset.shape)

    print("Class distributions is:")
    countClass = dataset.groupby('Label').size()
    print(countClass)

    print("Attribute's types are: ")
    print(dataset.dtypes)

    print("Correlations is: ")
    correlations = dataset.corr(method='pearson')
    print(correlations)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    plt.show()

    sns.scatterplot(data=dataset)
    plt.show()

    testDataset = dataset.loc[:, ["ST1.1", "ST2.1", "ST3.1"]]
    testDataset.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
    plt.show()

    dataset.plot(kind='density', subplots=True, layout=(42, 42), sharex=False)
    plt.show()


def classification(dataset):
    dataset.fillna(dataset.mean(), inplace=True)
    X = dataset.loc[:, 'Round':'ST5.2']
    y = dataset.loc[:, ['Label']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train, sample_weight=None, check_input=True, X_idx_sorted="deprecated")

    y_pred = clf.predict(X_test)

    result = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix: ")
    print(result)

    result1 = classification_report(y_test, y_pred)
    print("Classification Report: ")
    print(result1)

    result2 = accuracy_score(y_test, y_pred)
    print("Accuracy: ", result2)


def clustering(dataset):
    dataset.fillna(dataset.mean(), inplace=True)
    X = dataset.loc[:, 'Round':'ST5.2']

    kmeans = KMeans(n_clusters=8)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    plt.scatter(X.loc[:, ["ST1.1"]], X.loc[:, ["ST2.1"]], c=y_kmeans, s=20, cmap='summer')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=0.9)
    plt.show()


def RSME(y, y_pred):
    RSMESum = 0
    num = 0
    for i in range(len(y_pred)):
        RSMESum = pow(y_pred[i] - y.iloc[i], 2) + RSMESum
        num = num + 1

    RSME = math.sqrt(RSMESum / num)

    print("RSME is: ", RSME)

def regression(dataset):
    dataset.fillna(dataset.mean(), inplace=True)
    X = dataset.loc[:, 'Round':'ST4.2']
    y = dataset.loc[:, ['ST5.2']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    RSME(y, y_pred)
    plt.plot(y_pred)
    plt.scatter(y, y, s=20, c='red')
    plt.show()

    poly_reg = PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)

    y_pred_poly = pol_reg.predict(X_poly)

    RSME(y, y_pred_poly)
    plt.plot(y_pred_poly)
    plt.scatter(y, y, s=20, c='red')
    plt.show()


if __name__ == '__main__':
    dataset = loadData()
    analyzeData(dataset)
    classification(dataset)
    sampleDataset = dataset.sample(frac=0.3)
    classification(sampleDataset)
    clustering(dataset)
    regression(dataset)
    print("Done")
