#!/usr/bin/python
# -*- coding:utf-8 -*-

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
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from sklearn.externals import joblib
import pickle


def test_clf(name, clf, x_train, y_train):
    print ('Classifier：', clf)
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
    print ('Training time for 5 -fold cross validation：%.3f/(5*%d) = %.3fsec' % ((t_end - t_start), m, t_train))
    print( 'Optimal hyperparameter：', model.best_params_)
    joblib.dump(model, "%s.joblib"%name)
    t_start = time()
    y_hat = model.predict(x_test)
    t_end = time()
    t_test = t_end - t_start
    print ('Testing Time：%.3f sec' % t_test)
    acc = metrics.accuracy_score(y_test, y_hat)
    print ('Accuracy ：%.2f%%' % (100 * acc))
    name = str(clf).split('(')[0]
    index = name.find('Classifier')
    if index != -1:
        name = name[:index]
    if name == 'SVC':
        name = 'SVM'
    return t_train, t_test, 1-acc, name




def get_data():

    remove = ()

    categories = 'alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'

    print('start downloading...')

    t_start = time()

    data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=0, remove=remove)
    data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=0, remove=remove)

    t_end = time()

    print('downloading completed，take %.3f sec' % (t_end - t_start))

    return data_train, data_test


def print_data_info(data_train, data_test):

    print('data type：', type(data_train))
    print('# of texts in train set ：', len(data_train.data))
    print('# of texts in test set：', len(data_test.data))
    print('name of%d categories：' % len(data_train.target_names))

    pprint(data_train.target_names)


def get_y_data(data_train, data_test):

    y_train = data_train.target
    y_test = data_test.target

    return y_train, y_test


def tfidf_data(data_train, data_test):

    vectorizer = TfidfVectorizer(input='content', stop_words='english', max_df=0.5, sublinear_tf=True)

    vec = vectorizer.fit(data_train.data)
    pickle.dump(vec, open("vec.pickle", "wb"))
    x_train = vectorizer.transform(data_train.data)  # x_train is sparse，scipy.sparse.csr.csr_matrix
    x_test = vectorizer.transform(data_test.data)

    return x_train, x_test, vectorizer


def print_examples(y_train, data_train):

    print(' -- Examples : the first 10 texts -- ')

    categories = data_train.target_names

    for i in np.arange(10):
        print('category for text%d : %s' % (i + 1, categories[y_train[i]]))
        print(data_train.data[i])
        print('\n\n')


def print_x_data(x_train, vectorizer):

    print('# of train set：%d，# of features：%d' % x_train.shape)
    print('stop words:\n', )

    feature_names = np.asarray(vectorizer.get_feature_names())

    pprint(vectorizer.get_stop_words())

def classifier(x, y):

    print('\n\n===================\n evaluation of classifiers：\n')
    clfs = {"MultinomialNB": MultinomialNB(), 
            "BernoulliNB": BernoulliNB(),  
            "K_Neighbors": KNeighborsClassifier(),  
            "Ridge_Regression": RidgeClassifier(),  
            "RandomForest": RandomForestClassifier(n_estimators=200),  
            "SVC": SVC()  
            }
    result = []
    for name,clf in clfs.items():
        a = test_clf(name, clf, x, y)
        result.append(a)
        print('\n')
    return np.array(result)


def draw(result):
    time_train1, time_test1, err1, names = result.T
    time_test = time_test1.astype(np.float)
    time_train = time_train1.astype(np.float)
    err = err1.astype(np.float)
    x= np.arange(len(time_train))
    bar_width = 0.25
    ax1 = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right = 0.75)

    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    offset3 = 60
    offset2 = 0

    new_fixed_axis = ax3.get_grid_helper().new_fixed_axis
    ax3.axis["right"] = new_fixed_axis(loc="right", axes=ax3, offset=(offset3, 0))
    ax3.axis["right"].toggle(all=True)

    new_fixed_axis2 = ax2.get_grid_helper().new_fixed_axis
    ax2.axis["right"] = new_fixed_axis2(loc="right", axes=ax2, offset=(offset2, 0))
    ax2.axis["right"].toggle(all=True)

    ax1.set_ylabel("Error percentage")
    ax2.set_ylabel("Training time")
    ax3.set_ylabel("Testing time")

    b1 = ax1.bar(x, err, bar_width, alpha=0.2, color='r')
    b2 = ax2.bar(x + bar_width, time_train, bar_width, alpha=0.2, color='g')
    b3 = ax3.bar(x + bar_width * 2, time_test, bar_width, alpha=0.2, color='b')
    plt.xticks(x + bar_width * 2, names)
    plt.legend([b1[0], b2[0], b3[0]], ('Error Percentage', 'Training Time', 'Testing Time'), loc='upper left')
    plt.xlabel('Different Types Of Classifiers')
    plt.title('Evaluation Of Different Classifiers')
    plt.savefig("Performance_his.png")
    plt.show()

if __name__ == "__main__":

    data_train, data_test = get_data()
    print_data_info(data_train, data_test)

    y_train, y_test = get_y_data(data_train, data_test)
    print_examples(y_train, data_train)

    x_train, x_test, vectorizer = tfidf_data(data_train, data_test)
    print_x_data(x_train, vectorizer)

    draw(classifier(x_train, y_train))
