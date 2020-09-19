import matplotlib.pyplot as plt
from assignment1 import split_train_test_breast_cancer,split_train_test_spam
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
import numpy as np

def KNN_vs_k():
	train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
	k_list=[1,3,5,7,9,11,13,15,17,19]
	pipe=Pipeline([('scaler',StandardScaler()),('knn',KNeighborsClassifier())])
	acc_train_cancer_list=[]
	acc_test_cancer_list=[]
	for k in k_list:
		pipe.set_params(knn__n_neighbors=k)
		pipe.fit(train_features_cancer,train_labels_cancer)
		acc_train_cancer=pipe.score(train_features_cancer,train_labels_cancer)
		acc_test_cancer=pipe.score(test_features_cancer,test_labels_cancer)
		acc_train_cancer_list.append(acc_train_cancer)
		acc_test_cancer_list.append(acc_test_cancer)

	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	acc_train_spam_list=[]
	acc_test_spam_list=[]
	for k in k_list:
		pipe.set_params(knn__n_neighbors=k)
		pipe.fit(train_features_spam,train_labels_spam)
		acc_train_spam=pipe.score(train_features_spam,train_labels_spam)
		acc_test_spam=pipe.score(test_features_spam,test_labels_spam)
		acc_train_spam_list.append(acc_train_spam)
		acc_test_spam_list.append(acc_test_spam)	
	plt.figure(figsize=(10,6))
	plt.subplot(121)
	plt.plot(k_list,acc_train_cancer_list)
	plt.plot(k_list,acc_test_cancer_list)
	plt.xticks(k_list)
	plt.xlabel('k value')
	plt.ylabel('accuracy')
	plt.title('knn cancer classifier performance vs K')
	plt.subplot(122)
	plt.plot(k_list,acc_train_spam_list)
	plt.plot(k_list,acc_test_spam_list)
	plt.xticks(k_list)
	plt.xlabel('k value')
	plt.ylabel('accuracy')
	plt.title('knn spam classifier performance vs K')
	plt.show()

def KNN_vs_k_weighted():
	train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
	k_list=[1,3,5,7,9,11,13,15,17,19]
	pipe=Pipeline([('scaler',StandardScaler()),('knn',KNeighborsClassifier(weights='distance'))])
	acc_train_cancer_list=[]
	acc_test_cancer_list=[]
	for k in k_list:
		pipe.set_params(knn__n_neighbors=k)
		pipe.fit(train_features_cancer,train_labels_cancer)
		acc_train_cancer=pipe.score(train_features_cancer,train_labels_cancer)
		acc_test_cancer=pipe.score(test_features_cancer,test_labels_cancer)
		acc_train_cancer_list.append(acc_train_cancer)
		acc_test_cancer_list.append(acc_test_cancer)

	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	acc_train_spam_list=[]
	acc_test_spam_list=[]
	for k in k_list:
		pipe.set_params(knn__n_neighbors=k)
		pipe.fit(train_features_spam,train_labels_spam)
		acc_train_spam=pipe.score(train_features_spam,train_labels_spam)
		acc_test_spam=pipe.score(test_features_spam,test_labels_spam)
		acc_train_spam_list.append(acc_train_spam)
		acc_test_spam_list.append(acc_test_spam)	
	plt.figure(figsize=(10,6))
	plt.subplot(121)
	plt.plot(k_list,acc_train_cancer_list)
	plt.plot(k_list,acc_test_cancer_list)
	plt.xticks(k_list)
	plt.xlabel('k value')
	plt.ylabel('accuracy')
	plt.title('weighted knn cancer classifier performance vs K')
	plt.subplot(122)
	plt.plot(k_list,acc_train_spam_list)
	plt.plot(k_list,acc_test_spam_list)
	plt.xticks(k_list)
	plt.xlabel('k value')
	plt.ylabel('accuracy')
	plt.title('weighted knn spam classifier performance vs K')
	plt.show()


def radius_neighbors_r():
	train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
	r_list=[2**0,2**1,2**2,2**3,2**4,2**5,2**6,2**7,2**8,2**9,2**10]
	pipe=Pipeline([('scaler',StandardScaler()),('rn',RadiusNeighborsClassifier(outlier_label='most_frequent'))])
	acc_train_cancer_list=[]
	acc_test_cancer_list=[]
	for r in r_list:
		pipe.set_params(rn__radius=r)
		pipe.fit(train_features_cancer,train_labels_cancer)
		acc_train_cancer=pipe.score(train_features_cancer,train_labels_cancer)
		acc_test_cancer=pipe.score(test_features_cancer,test_labels_cancer)
		acc_train_cancer_list.append(acc_train_cancer)
		acc_test_cancer_list.append(acc_test_cancer)

	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	acc_train_spam_list=[]
	acc_test_spam_list=[]
	for r in r_list:
		pipe.set_params(rn__radius=r)
		pipe.fit(train_features_spam,train_labels_spam)
		acc_train_spam=pipe.score(train_features_spam,train_labels_spam)
		acc_test_spam=pipe.score(test_features_spam,test_labels_spam)
		acc_train_spam_list.append(acc_train_spam)
		acc_test_spam_list.append(acc_test_spam)	
	plt.figure(figsize=(10,6))
	plt.subplot(121)
	plt.plot(r_list,acc_train_cancer_list)
	plt.plot(r_list,acc_test_cancer_list)
	plt.xscale('log')
	plt.xlabel('radius value')
	plt.ylabel('accuracy')
	plt.title('radius neighbor cancer \nclassifier performance vs K')
	plt.subplot(122)
	plt.plot(r_list,acc_train_spam_list)
	plt.plot(r_list,acc_test_spam_list)
	plt.xscale('log')
	plt.xlabel('radius value')
	plt.ylabel('accuracy')
	plt.title('radius neighbor spam \nclassifier performance vs K')
	plt.show()

def learning_curve():
	pipe=Pipeline([('scaler',StandardScaler()),('knn',KNeighborsClassifier())])
	num_cancer=[50,100,150,200,250,300,350,400,450,500]
	num_spam=[400,800,1200,1600,2000,2400,2800,3200,3600,4000]
	train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
	acc_train_cancer_list=[]
	acc_test_cancer_list=[]
	for num in num_cancer:
		pipe.set_params(knn__n_neighbors=3)
		pipe.fit(train_features_cancer[:num,:],train_labels_cancer[:num])
		acc_train_cancer=pipe.score(train_features_cancer[:num,:],train_labels_cancer[:num])
		acc_test_cancer=pipe.score(test_features_cancer,test_labels_cancer)
		acc_train_cancer_list.append(acc_train_cancer)
		acc_test_cancer_list.append(acc_test_cancer)

	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	acc_train_spam_list=[]
	acc_test_spam_list=[]
	for num in num_spam:
		pipe.set_params(knn__n_neighbors=3)
		pipe.fit(train_features_spam[:num,:],train_labels_spam[:num])
		acc_train_spam=pipe.score(train_features_spam[:num,:],train_labels_spam[:num])
		acc_test_spam=pipe.score(test_features_spam,test_labels_spam)
		acc_train_spam_list.append(acc_train_spam)
		acc_test_spam_list.append(acc_test_spam)

	plt.figure(figsize=(10,6))
	plt.subplot(121)
	plt.plot(num_cancer,acc_train_cancer_list)
	plt.plot(num_cancer,acc_test_cancer_list)
	plt.xlabel('training size')
	plt.ylabel('accuracy')
	plt.title('knn cancer classifier \nperformance vs training size')
	plt.subplot(122)
	plt.plot(num_spam,acc_train_spam_list)
	plt.plot(num_spam,acc_test_spam_list)
	plt.xlabel('training size')
	plt.ylabel('accuracy')
	plt.title('knn spam classifier \nperformance vs training size')
	plt.show()


def gridSearch():
	train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
	pipe=Pipeline([('scaler',StandardScaler()),('knn',KNeighborsClassifier())])
	k_list=[1,3,5,7,9,11,13,15,17,19]
	clf=GridSearchCV(pipe,[{'knn__n_neighbors':k_list}],cv=5)
	clf.fit(train_features_cancer,train_labels_cancer)
	best_train_acc_cancer=clf.score(train_features_cancer,train_labels_cancer)
	best_test_acc_cancer=clf.score(test_features_cancer,test_labels_cancer)
	print('best train accuracy for cancer knn classifier is:',best_train_acc_cancer)
	print('best test accuracy for cancer knn classifier is:',best_test_acc_cancer)
	print('best hyperparameters for cancer knn classifier is:',clf.best_params_)
	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	clf.fit(train_features_spam,train_labels_spam)
	best_train_acc_spam=clf.score(train_features_spam,train_labels_spam)
	best_test_acc_spam=clf.score(test_features_spam,test_labels_spam)
	print('------------------------------------------------')
	print('best train accuracy for spam knn classifier is:',best_train_acc_spam)
	print('best test accuracy for spam knn classifier is:',best_test_acc_spam)
	print('best hyperparameters for spam knn classifier is:',clf.best_params_)



if __name__=="__main__":
	KNN_vs_k()
	KNN_vs_k_weighted()
	radius_neighbors_r()
	learning_curve()
	gridSearch()


