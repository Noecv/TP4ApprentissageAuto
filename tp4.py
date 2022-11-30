import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


reader = csv.reader(open("WineQT.csv", "r"), delimiter=",")
x = np.array(list(reader))
#x = x[0:500] # permet de réduire la taille du jeu de données pour les tests
x = np.delete(x, -1, axis=1)
idcolumn = x[0]
x = np.delete(x, 0, axis=0)
y = x[:,-1]
x = np.delete(x, -1, axis=1)

sc = StandardScaler()
x = sc.fit_transform(x)

y = y.astype(float)


# Création des jeux de donnée d'entrainement
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# creating an object of LinearRegression class
LR = LinearRegression()
# fitting the training data
LR.fit(x_train, y_train)

y_prediction = LR.predict(x_test)
#print(y_prediction)

score=r2_score(y_test,y_prediction)
coefs = LR.coef_
print('r2 score is', score)
print('coefs', coefs)
print(idcolumn)


a = x[:,-1]
b = x[:,1]
c = x[:,6]



new_x = np.concatenate(([a],[b],[c]))
print("shape", len(new_x.T), new_x.T[0])
WCSS = []
size = 20
for i in range(1,20):
    model = KMeans(n_clusters = i, init = 'k-means++')
    model.fit(new_x.T)
    WCSS.append(model.inertia_)
fig = plt.figure(figsize = (7,7))
plt.plot(range(1,20),WCSS, linewidth=4, markersize=12,marker='o',color = 'green')
plt.xticks(np.arange(20))
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show() #elbow at 3


model = KMeans(n_clusters = 4, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_clusters = model.fit_predict(new_x.T)
print(len(new_x), len(new_x[0]))
print(y_clusters)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
color = ['yellow','green','red', 'orange']
for i in range(len(new_x[0])):
    ax.scatter(new_x[0][i], new_x[1][i], new_x[2][i], lw=0.2, marker='o', c = color[y_clusters[i]])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
#color1 = ['black','dimgray','dimgrey','gray', 'grey', 'darkgray', 'silver', 'lightgray', 'gainsboro', 'whitesmoke']
#for i in range(len(a)):
#    ax.scatter(new_x[0][i], new_x[1][i], new_x[2][i], marker='o', c = color1[int(y[i])])

color1 = ['red','orange','yellow', 'green']
indicebis = [0,0,0,0,0,1,2,3,3,3]
for i in range(len(a)):
    ax.scatter(new_x[0][i], new_x[1][i], new_x[2][i], marker='o', c = color1[indicebis[int(y[i])]])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show() 


#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#plt.scatter(new_x.reshape(-1,1)[y_clusters == 0,0],new_x.reshape(-1,1)[y_clusters == 0,1],new_x.reshape(-1,1)[y_clusters == 0,2],s = 50, c = 'green', label = "")
#plt.scatter(new_x.reshape(-1,1)[y_clusters == 1,0],new_x.reshape(-1,1)[y_clusters == 1,1],new_x.reshape(-1,1)[y_clusters == 1,2],s = 50, c = 'blue', label = "")
#plt.scatter(new_x.reshape(-1,1)[y_clusters == 2,0],new_x.reshape(-1,1)[y_clusters == 2,1],new_x.reshape(-1,1)[y_clusters == 2,2],s = 50, c = 'red', label = " ")
#plt.xlabel("Anual income(k$) -- >")
#plt.ylabel("spending score out of 100 -- >")
#plt.legend()
#plt.show()



