#!/usr/bin/env python
# coding: utf-8
# https://www.analyticsvidhya.com/blog/2017/09/machine-learning-models-as-apis-using-flask/
# https://towardsdatascience.com/create-an-api-to-deploy-machine-learning-models-using-flask-and-heroku-67a011800c50


import os
import pickle
import json
import keras
import requests
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from fastai.tabular import *
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from flask import Flask, jsonify, request
from pandas.plotting import scatter_matrix

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
pd.set_option('mode.chained_assignment', None)

path = Path('/Users/bbabu/fastai/DataSamples')


# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    df = pd.read_json(data, orient='records')

    df.drop(['Description', 'MemberDN', 'State', 'Owner'], axis=1, inplace=True)
    df.drop_duplicates(keep='first',inplace=True) 

    df_2 = df.copy()
    
    grouped = df_2.groupby(['GroupDN'], as_index=False)

    for name, group in grouped:
        df_group = group.copy()
        categorical = list(group.select_dtypes(include=['object']).columns.values)
        for cat in categorical:
            le = preprocessing.LabelEncoder()
            group[cat].fillna('NaN', inplace=True)
            group[cat] = le.fit_transform(group[cat].astype(str))
        filename = path/'Final_Model-NN-V5.sav'
        model = pickle.load(open(filename, 'rb'))
        sc = StandardScaler()
        X_test = group
        X_test = sc.fit_transform(X_test)
        pred = model.predict(X_test)
        final_df = pd.DataFrame({'GroupDN': df_group['GroupDN'],'Member': df_group['Member'], 'Manager': df_group['Manager'], 'Title': df_group['Title'], 'Platform': df_group['Platform'], 'City': df_group['City'], 'Owner': pred[:, 0]})
        out = final_df.loc[final_df['Owner'].idxmax()]
        final_df.sort_values(['Owner'], inplace=True, ascending=False)
        final_df.to_csv(path/"".join([(out[0].split(',')[0].replace("CN=","")),'.csv']), header=True, index=False)
        output = print("Predicated Owner for the group %s is %s with designation %s"%((out[0].split(',')[0].replace("CN=","")), out[1], out[3]))

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=False)


