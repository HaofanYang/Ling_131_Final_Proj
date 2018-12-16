#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from time import time
from pprint import pprint
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib as mpl


def test_clf(clf):
    print (u'Classifier：', clf)
    alpha_can = np.logspace(-3, 2, 10)
    model = GridSearchCV(clf, param_grid={'alpha': alpha_can}, cv=5)
    m = alpha_can.size
    if hasattr(clf, 'alpha'):
        model.set_params(param_grid={'alpha': alpha_can})
        m = alpha_can.size
    if hasattr(clf, 'n_neighbors'):
        neighbors_can = np.arange(1, 15)
        model.set_params(param_grid={'n_neighbors': neighbors_can})
        m = neighbors_can.size
    if hasattr(clf, 'C'):
        C_can = np.logspace(1, 3, 3)
        gamma_can = np.logspace(-3, 0, 3)
        model.set_params(param_grid={'C':C_can, 'gamma':gamma_can})
        m = C_can.size * gamma_can.size
    if hasattr(clf, 'max_depth'):
        max_depth_can = np.arange(4, 10)
        model.set_params(param_grid={'max_depth': max_depth_can})
        m = max_depth_can.size
    t_start = time()
    model.fit(x_train, y_train)
    t_end = time()
    t_train = (t_end - t_start) / (5*m)
    print (u'Training time for 5 -fold cross validation：%.3f/(5*%d) = %.3fsec' % ((t_end - t_start), m, t_train))
    print( u'Optimal hyperparameter：', model.best_params_)
    t_start = time()
    y_hat = model.predict(x_test)
    t_end = time()
    t_test = t_end - t_start
    print (u'Testing Time：%.3f sec' % t_test)
    acc = metrics.accuracy_score(y_test, y_hat)
    print (u'Accuracy ：%.2f%%' % (100 * acc))
    name = str(clf).split('(')[0]
    index = name.find('Classifier')
    if index != -1:
        name = name[:index]     # 去掉末尾的Classifier
    if name == 'SVC':
        name = 'SVM'
    return t_train, t_test, 1-acc, name




def get_data():

    #remove = ('headers', 'footers', 'quotes')

    remove = ()

    categories = 'alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'

    print(u'start downloading...')

    t_start = time()

    data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=0, remove=remove)
    data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=0, remove=remove)

    t_end = time()

    print(u'downloading completed，take %.3f sec' % (t_end - t_start))

    return data_train, data_test


def print_data_info(data_train, data_test):

    print(u'data type：', type(data_train))
    print(u'# of texts in train set ：', len(data_train.data))
    print(u'# of texts in test set：', len(data_test.data))
    print(u'name of%d categories：' % len(data_train.target_names))

    pprint(data_train.target_names)


def get_y_data(data_train, data_test):

    y_train = data_train.target
    y_test = data_test.target

    return y_train, y_test


def tfidf_data(data_train, data_test):

    vectorizer = TfidfVectorizer(input='content', stop_words='english', max_df=0.5, sublinear_tf=True)

    x_train = vectorizer.fit_transform(data_train.data)  # x_train是稀疏的，scipy.sparse.csr.csr_matrix
    x_test = vectorizer.transform(data_test.data)

    return x_train, x_test, vectorizer


def print_examples(y_train, data_train):

    print(u' -- Examples : the first 10 texts -- ')

    categories = data_train.target_names

    for i in np.arange(10):
        print(u'category for text%d : %s' % (i + 1, categories[y_train[i]]))
        print(data_train.data[i])
        print('\n\n')


def print_x_data(x_train, vectorizer):

    print(u'# of train set：%d，# of features：%d' % x_train.shape)
    print(u'stop words:\n', )

    pprint(vectorizer.get_stop_words())
    feature_names = np.asarray(vectorizer.get_feature_names())






if __name__ == "__main__":

    data_train, data_test = get_data()

    print_data_info(data_train, data_test)

    y_train, y_test = get_y_data(data_train, data_test)

    print_examples(y_train, data_train)

    x_train, x_test, vectorizer = tfidf_data(data_train, data_test)

    print_x_data(x_train, vectorizer)




    print (u'\n\n===================\n evaluation of classifiers：\n')
    clfs = (MultinomialNB(),                # 0.87(0.017), 0.002, 90.39%
            BernoulliNB(),                  # 1.592(0.032), 0.010, 88.54%
            KNeighborsClassifier(),         # 19.737(0.282), 0.208, 86.03%
            RidgeClassifier(),              # 25.6(0.512), 0.003, 89.73%
            RandomForestClassifier(n_estimators=200),   # 59.319(1.977), 0.248, 77.01%
            SVC()                           # 236.59(5.258), 1.574, 90.10%
            )
    result = []
    for clf in clfs:
        a = test_clf(clf)
        result.append(a)
        print ('\n')
    result = np.array(result)
    time_train, time_test, err, names = result.T
    x = np.arange(len(time_train))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 7), facecolor='w')
    ax = plt.axes()
    b1 = ax.bar(x, err, width=0.25, color='#77E0A0')
    ax_t = ax.twinx()
    b2 = ax_t.bar(x+0.25, time_train, width=0.25, color='#FFA0A0')
    b3 = ax_t.bar(x+0.5, time_test, width=0.25, color='#FF8080')
    plt.xticks(x+0.5, names, fontsize=10)
    leg = plt.legend([b1[0], b2[0], b3[0]], (u'error percentage', u'training time', u'testing time'), loc='upper left', shadow=True)
    # for lt in leg.get_texts():
    #     lt.set_fontsize(14)
    plt.title(u'evaluation of different classifiers', fontsize=18)
    plt.xlabel(u'name of classifier')
    plt.grid(True)
    plt.tight_layout(2)
    plt.show()
