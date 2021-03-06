#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
	>>> python3 run.py
	The program runs classifiers and vectorizor trained and saved by train.py, prints predicted category of file test.txt.
	Thus we need to run the train.py first, as we need the classifiers and vectorizor in the first place.
'''

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def run(x, clf_name):
	clf_name = clf_name + ".joblib"
	clf = joblib.load(clf_name)
	return clf.predict(x)

def get_tfidf(filename):
	text = ""
	vectorizer = pickle.load(open("vec.pickle", 'rb'))
	with open(filename) as fl:
		for line in fl:
			text += line
	return vectorizer.transform([text])


if __name__ == "__main__":
	run_file = "test.txt" # Please specify the file name you want to test
	x = get_tfidf(run_file)
	print("The classification for %s is:" % run_file)
	clfs = ("MultinomialNB", "BernoulliNB",  "K_Neighbors",  "Ridge_Regression",  "RandomForest",  "SVC")
	cat = ('comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',\
                 'comp.windows.x', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey')
	for clf in clfs:
		print("%s: %s\n" % (clf, cat[run(x, clf)[0]]))