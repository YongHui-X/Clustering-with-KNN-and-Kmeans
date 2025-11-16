import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

'''Display Histograms'''
def plot_distr(df):
    sns.set(style='darkgrid')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
    axes = axes.flatten()

    cols = ['Age', 'Income', 'Years Employed', 'DebtIncomeRatio']

    for i, col in enumerate(cols):
        sns.histplot(
            data=df, 
            x=col, 
            hue='Defaulted', 
            kde=True, 
            ax=axes[i], 
            palette='tab10', 
            alpha=0.6  # controls shading transparency
        )
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
    
    plt.tight_layout(pad=3.0) #padding between plots
    plt.show(block=True)

'''Plot number of defaults and non-defaults'''
def plot_defaults(df):
    fig, ax = plt.subplots()

    #string to handle data
    df['Defaulted'] = df['Defaulted'].astype(str)
    sns.histplot(df, x='Defaulted', hue='Defaulted', palette='tab10')


    plt.title('Credit Card Defaulted Cases')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Non-Default', 'Defaulted'])
    plt.xlabel('')
    plt.ylabel('No. of People')
    plt.show(block=True)

'''Plot Clustering Results'''

def plot_cluster_results(df, clusters):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    x = df['Age']
    y = df['Income']
    z = df['DebtIncomeRatio']

    ax.scatter(x, y, z, c=clusters, cmap='viridis', edgecolor='k', linewidth=0.5) #ax allows 3d unlike seaborn 2d

    ax.set_xlabel('Age')
    ax.set_ylabel('Income')
    ax.set_zlabel('DebtIncomeRatio')
    ax.set_title('3D K-Means Clustering of Customers')

    plt.show()

'''Main'''
df = pd.read_csv("cust_seg.csv")
df.drop(["Customer Id"], axis=1, inplace=True)
df.columns = df.columns.str.strip()
#plots  4 histograms
plot_distr(df)

'''KNN'''
#classify the missing Defaulted values using KNN

#seperate into 2 df
df_known = df[df['Defaulted'].notna()] #True where value exists (0 or 1)

df_missing = df[df['Defaulted'].isna()] #True where the value is missing

x_known = df_known.drop('Defaulted', axis=1) #dont want defaulted col
y_known = df_known['Defaulted']

#Prediction
x_missing = df_missing.drop('Defaulted', axis=1)

'''Scaler on x_known'''

scaler = StandardScaler()
x_known_scaled = scaler.fit_transform(x_known) #fit only known rows,
x_missing_scaled = scaler.transform(x_missing) #transform missing rows

#create knn model
knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(x_known_scaled,y_known)

#Predict Defaulted values for missing rows
y_pred = knn_model.predict(x_missing_scaled)

#insert back, Take all the rows where Defaulted is missing, and fill them with predictions
df.loc[df["Defaulted"].isna(), 'Defaulted'] = y_pred #returns a boolean mask: True where Defaulted is missing, False otherwise.

#print(df.head(10))

#plots defaults
plot_defaults(df)

'''K Means clustering'''
#print("Start K means Clustering")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#gets kmeans obj
algo = KMeans(n_clusters=4)

#gives cluster-assignment for each row
clusters = algo.fit_predict(scaled_data)

print('Clustering results: ', clusters)

plot_cluster_results(df,clusters)
