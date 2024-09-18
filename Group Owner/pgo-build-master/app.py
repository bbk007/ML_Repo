import os
import random
import urllib.request
from flask import Flask, jsonify, flash, request, redirect, render_template, request

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
from pandas.plotting import scatter_matrix

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
pd.set_option('mode.chained_assignment', None)

TMP_DIR = '/var/tmp/'
app = Flask(__name__)

app.secret_key = "89wrjns8wefnvdiweu"
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['.csv'])


def process_csv(file):
    # convert data into dataframe
    #df = pd.read_json(file, orient='records')
    df = pd.read_csv(file)

    df.drop(['Description', 'MemberDN', 'State', 'Owner'], axis=1, inplace=True)
    df.drop_duplicates(keep='first', inplace=True)

    df_2 = df.copy()

    grouped = df_2.groupby(['GroupDN'], as_index=False)
    results = []

    for name, group in grouped:
        df_group = group.copy()
        
        categorical = list(group.select_dtypes(include=['object']).columns.values)
        
        for cat in categorical:
            le = preprocessing.LabelEncoder()
            group[cat].fillna('NaN', inplace=True)
            group[cat] = le.fit_transform(group[cat].astype(str))

        model = pickle.load(open('res/Final_Model-NN-V6.sav', 'rb'))
        sc = StandardScaler()
        X_test = group
        X_test = sc.fit_transform(X_test)
        pred = model.predict(X_test)
        final_df = pd.DataFrame({'GroupDN': df_group['GroupDN'], 'Member': df_group['Member'], 'Manager': df_group['Manager'],
                                 'Title': df_group['Title'], 'Platform': df_group['Platform'], 'City': df_group['City'], 'Owner': pred[:, 0]})

        out = final_df.loc[final_df['Owner'].idxmax()]
        final_df.sort_values(['Owner'], inplace=True, ascending=False)

        name = out[0].split(',')[0].replace("CN=", "")

        final_df.to_csv(TMP_DIR +name +'.csv', header=True, index=False)

        result = {}
        result['group'] = name
        result['owner'] = out[1]
        result['designation'] = out[3]
        results.append(result)
        #results.append(print("Predicated Owner for the group %s is %s with designation %s" % (name, out[1], out[3])))

    # return data
    return jsonify(results=results)


def allowed_file(filename):
    return get_ext(filename) in ALLOWED_EXTENSIONS


def get_ext(filename):
    name, ext = os.path.splitext(filename)
    return ext


@app.route('/upload', methods=['POST'])
def upload_file():

    print("Got upload request!")

    if 'file' not in request.files:
        print("No files in request")
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        print("Filename is null")
        flash('No file selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        print("Processing upload...")
        #filename = file.filename + str(random.randint(1, 1000000))
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return process_csv(file)
    else:
        print("File type not allowed")
        flash('Only CSV files are allowed!')
        return redirect(request.url)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run()



