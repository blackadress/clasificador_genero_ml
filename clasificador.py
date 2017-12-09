from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#[altura, peso, talla de zapato]
x_train = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
	 	  [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
	 	  [159, 55, 37],[171, 75, 42], [181, 85, 43]]
#Para mejores resultados definir un x_test diferente a x_train

y_train = ['varon', 'varon', 'mujer', 'mujer', 
	 	   'varon', 'varon', 'mujer', 'mujer',
	 	   'mujer','varon', 'varon']
#Para mejores resultados definir un y_test diferente a t_train

#Definiendo los clasificadores
clf_tree = tree.DecisionTreeClassifier()
clf_svm = svm.SVC()
clf_KNN = KNeighborsClassifier()
clf_gauss = GaussianNB()

#Entrenando los modelos
clf_tree.fit(x_train, y_train)
clf_svm.fit(x_train, y_train)
clf_KNN.fit(x_train, y_train)
clf_gauss.fit(x_train, y_train)

#predicciones y certeza del modelo

pred_tree = clf_tree.predict(x_train)
acc_tree = accuracy_score(y_train, pred_tree)
print('La prediccion con arbol de decisiones es: {}, con una certeza de: {}.'.
		format(pred_tree, acc_tree))

pred_svm = clf_tree.predict(x_train)
acc_svm = accuracy_score(y_train, pred_svm)
print('La prediccion con SVM es: {}, con una certeza de: {}.'.
		format(pred_svm, acc_svm))

pred_KNN = clf_KNN.predict(x_train)
acc_KNN = accuracy_score(y_train, pred_KNN)
print('La prediccion con Vecinos más cercanos K es: {}, con una certeza de: {}.'.
		format(pred_KNN, acc_KNN))

pred_gauss = clf_gauss.predict(x_train)
acc_gauss = accuracy_score(y_train, pred_gauss)
print('La prediccion con Gauss ingenuo es: {}, con una certeza de: {}.'.
		format(pred_gauss, acc_gauss))
