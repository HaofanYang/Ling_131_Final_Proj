
#!/usr/bin/python
# -*- coding:utf-8 -*-
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

def run(x, clf_name):
	clf_name = clf_name + ".joblib"
	clf = joblib.load(clf_name)
	return clf.predict(x)

def get_tfidf(filename):
	text = ""
	vectorizer = pickle.load(open("vec.pickle"))
	with open(filename) as fl:
		for line in fl:
			text += line
	return vectorizer.transform([text])  


if __name__ == "__main__":
	run_file = "test.txt" # Please specify the file name you want to test
	x = get_tfidf(run_file)
	print("The classification for %s is:" % run_file)
	clfs = ["MultinomialNB", "BernoulliNB",  "K_Neighbors",  "Ridge_Regression",  "RandomForest",  "SVC"]
	for clf in clfs:
		print("%s: %s\n" % (clf, run(x, clf)))

