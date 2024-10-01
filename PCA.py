
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "pca_iris.data"

df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

df

features = ['sepal length', 'sepal width', 'petal length', 'petal width']

x = df[features]

y = df[['target']]


x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

final_dataframe = pd.concat([principalDf, df[['target']]], axis = 1)

final_dataframe.head()

dfsetosa= final_dataframe[df.target=='Iris-setosa']
dfvirginica = final_dataframe[df.target=='Iris-virginica']
dfversicolor = final_dataframe[df.target=='Iris-versicolor']
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')

plt.scatter(dfsetosa['principal component 1'], dfsetosa['principal component 2'],color='green')
plt.scatter(dfvirginica['principal component 1'], dfvirginica['principal component 2'],color='red')
plt.scatter(dfversicolor['principal component 1'], dfversicolor['principal component 2'],color='blue')

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['g', 'b', 'r']

plt.xlabel('principal component 1')
plt.ylabel('principal component 2')

for target, col in zip(targets,colors):
    dftemp = final_dataframe[df.target==target]
    plt.scatter(dftemp['principal component 1'], dftemp['principal component 2'], color=col)

pca.explained_variance_ratio_

pca.explained_variance_ratio_.sum()
