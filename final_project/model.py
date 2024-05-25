from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

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
    plt.scatter(df['time'][-window-1:], df['thres'][-window-1:], c=labels)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title('KMeans clustering using both value and slope as features')
    
    # print the time that the value is classified change
    for i in range(len(labels)-1):
        if labels[i] != labels[i+1]:
            print(df['time'][i])
    plt.show()
    return kmeans_model
    
# Agglomerative Hierarchical Clustering
def Hierarchical_Clustering_model(df, file_name, window = 15, realtime = True):
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
    hierarchical_model= AgglomerativeClustering(n_clusters=k)
    hierarchical_model.fit_predict(rolling_features)
    
    # draw the clustering result given the features and the time
    # get the cluster centers
    # centers = agglomerativeclustering_model.cluster_centers_
    # get the cluster labels
    labels = hierarchical_model.labels_
    # plot the data
    plt.scatter(df['time'][-window-1:], df['thres'][-window-1:], c=labels)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title('Agglomerative Clustering using both value and slope as features')
    
    # print the time that the value is classified change
    for i in range(len(labels)-1):
        if labels[i] != labels[i+1]:
            print(df['time'][i])
    plt.show()
    return hierarchical_model, labels

#Gaussian Mixture Models
def GaussianMixture_model(df, file_name, window = 15, realtime = True):
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
    k=2
    gmm = GaussianMixture(n_components=k)
    gmm.fit(rolling_features)
    # draw the clustering result given the features and the time
    # get the cluster centers
    # centers = agglomerativeclustering_model.cluster_centers_
    # get the cluster labels
    labels = hierarchical_model.labels_
    # plot the data
    plt.scatter(df['time'][-window-1:], df['thres'][-window-1:], c=labels)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title('Gaussian Mixture using both value and slope as features')
    
    # print the time that the value is classified change
    for i in range(len(labels)-1):
        if labels[i] != labels[i+1]:
            print(df['time'][i])
    plt.show()
    return gmm

# import random

# save_name = 'subject1.csv'

# df=pd.DataFrame()
# Thres = np.zeros(30)
# Slope = np.zeros(30)
# Time = np.zeros(30)

# for i in range(30):
#     Thres[i]=random.randint(1,100)
#     Slope[i]=random.uniform(-1,1)
#     Time[i]=i
# df = pd.DataFrame({'thres': Thres, 'slope': Slope, 'time': Time}) 
    
# print(df)


# kmeans_model=Kmeans_model(df, save_name, window = 15)
# hierarchical_model,label2=Hierarchical_Clustering_model(df, save_name, window = 15)
# gmm=GaussianMixture_model(df, save_name, window = 15)

# df=pd.DataFrame()
# Thres = Thres[-16:]
# Slope = Slope[-16:]

# # label2=agglomerativeclustering_model.fit_predict(df)
# df = pd.DataFrame({'thres': Thres, 'slope': Slope}) 
# label1=kmeans_model.predict(df)
# print(label1)
# print(label2)
# label3=gmm.predict(df)
# print(label3)