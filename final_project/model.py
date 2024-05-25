from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def Kmeans_model(df, file_name, window = 15, realtime = True):
    if not realtime:
        # read data from file
        pass
        """
        TODO read data from file
        """
    else:
        # open data folder and save the data
        if not os.path.exists('data'):
            os.makedirs('data')
        file_name = os.path.join('data', file_name)   
        df.to_csv(file_name, index=False)

    features = df[['thres', 'slope']]
    rolling_features = features.rolling(window).mean().dropna()
    # sclar = StandardScaler()
    # features = sclar.fit_transform(rolling_features)
    k = 2
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(rolling_features)
    
    # draw the clustering result given the features and the time
    # get the cluster centers
    centers = kmeans_model.cluster_centers_
    # get the cluster labels
    labels = kmeans_model.labels_
    # plot the data
    plt.scatter(df['time'], df['thres'], c=labels)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title('KMeans clustering using both value and slope as features')
    
    # print the time that the value is classified change
    for i in range(len(labels)-1):
        if labels[i] != labels[i+1]:
            print(df['time'][i])
    plt.show()
    return kmeans_model
    

    
   
    
    