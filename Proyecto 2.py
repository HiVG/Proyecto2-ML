# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:21:06 2020

@author: HP
"""
import pandas as pd
import numpy as np
import nltk
import csv

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import array as arr
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn import model_selection,svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


def load():
    X_train = pd.read_csv("train_data.csv", header=None)
    X_test = pd.read_csv("test_data.csv", header=None)
    
    # print (x_train)
    # print (X_train)    
    for item in X_train[0]:
        X_train[0]=X_train[0].replace(item,item.lower())
        
    for item in X_test[0]:
        X_test[0]=X_test[0].replace(item,item.lower())
   
    X_train[0]=[word_tokenize(item) for item in X_train[0]]
    X_test[0]=[word_tokenize(item) for item in X_test[0]]

    
    # #Data a usar
    # x_train = X_train[0]
    # x_test = X_test[0]
    # y_test = X_test[1]
    # y_train = X_train[1]
    
    tag_map=defaultdict(lambda:wn.NOUN)
    tag_map['J']=wn.ADJ
    tag_map['V']=wn.VERB
    tag_map['R']=wn.ADV
    
    for index,item in enumerate(X_train[0]):
        Final_words=[]
        word_Lemmatized=WordNetLemmatizer()
        for word,tag in pos_tag(item):
            if word not in stopwords.words('english') and word.isalpha():
                word_Final=word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        X_train.loc[index,'text_final']=str(Final_words)
    #print (X_train)
    
    for index,item in enumerate(X_test[0]):
        Final_words=[]
        word_Lemmatized=WordNetLemmatizer()
        for word,tag in pos_tag(item):
            if word not in stopwords.words('english') and word.isalpha():
                word_Final=word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        X_test.loc[index,'text_final']=str(Final_words)
    #print (X_train)
    
    x_train = X_train['text_final']
    x_test = X_test['text_final']
    y_test = X_test[1]
    y_train = X_train[1]
    
    modelSVM(X_train, X_test, x_train, x_test, y_train, y_test)
    rl(X_train, X_test, x_train, x_test, y_train, y_test)
    
def rl(X_train, X_test, x_train, x_test, y_train, y_test):
    enc=LabelEncoder()
    train_y=enc.fit_transform(y_train)
    test_y=enc.fit_transform(y_test)
    
    tfidf_vector=TfidfVectorizer(max_features=5000)
    tfidf_vector.fit(X_train['text_final'])
    Xtrain_tfidf=tfidf_vector.transform(x_train)
    Xtest_tfidf=tfidf_vector.transform(x_test)
    
    print("------Regresión Logística------")
    regresion = LogisticRegression(C=1.0, solver= 'liblinear', penalty= 'l1', random_state=1) # con lasso
    regresion.fit(Xtrain_tfidf,train_y)
    pred=regresion.predict(Xtest_tfidf)
    pred=enc.fit_transform(pred)
    print ("x_test")
    print(x_test)
    print(pred)
    print(accuracy_score(pred,test_y))
    print("----------métricas---------")
    print(classification_report(test_y, pred))
    
    grid = gridS(Xtrain_tfidf,train_y,regresion,"rl") 
    print(grid.best_params_)
    print(pd.DataFrame(grid.cv_results_))

def modelSVM(X_train, X_test, x_train, x_test, y_train, y_test):
    
    enc=LabelEncoder()
    train_y=enc.fit_transform(y_train)
    test_y=enc.fit_transform(y_test)
    
    tfidf_vector=TfidfVectorizer(max_features=5000)
    tfidf_vector.fit(X_train['text_final'])
    Xtrain_tfidf=tfidf_vector.transform(x_train)
    Xtest_tfidf=tfidf_vector.transform(x_test)
    
    print("------Maquina de Vector de Soporte-------")
    svmc=svm.SVC(C=1.0,kernel='linear',degree=3, gamma=-2)
    svmc.fit(Xtrain_tfidf,train_y)
    pred=svmc.predict(Xtest_tfidf)
    pred=enc.fit_transform(pred)
    print ("x_test")
    print(x_test)
    print(pred)
    print(accuracy_score(pred,test_y))
    print("----------métricas---------")
    print(classification_report(test_y, pred))
    
    
    grid = gridS(Xtrain_tfidf,train_y,svmc,"svm") 
    print(grid.best_params_)
    print(pd.DataFrame(grid.cv_results_))
    

def gridS(x,y,model,nombre):
    if nombre == "svm":
        params={"kernel":["linear","rbf"],
                "C": [0,2,3], "gamma":[-3,-2,2]}
    elif nombre == "rl":
        params={"penalty": ["l1","l2"],
                "solver":["newton-cg","lbfgs","liblinear"],
                "C":[0,2,3]}
    grid = GridSearchCV(estimator=model, param_grid=params, cv=5)
    grid.fit(x,y)
    return grid
    
    

def main():
    load()
    

if __name__=="__main__":
    main()