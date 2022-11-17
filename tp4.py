import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler





reader = csv.reader(open("WineQT.csv", "r"), delimiter=",")
x = np.array(list(reader))


x = np.delete(x, -1, axis=1)
idcolumn = x[0]
x = np.delete(x, 0, axis=0)
y = x[:,-1]
x = np.delete(x, -1, axis=1)

sc = StandardScaler()
x = sc.fit_transform(x)

print(x,y)

# Création des jeux de donnée d'entrainement
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print("\n########### K means ############")
