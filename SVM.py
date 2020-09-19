import matplotlib.pyplot as plt
from assignment1 import split_train_test_breast_cancer,split_train_test_spam
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np


num_cancer=[50,100,150,200,250,300,350,400,450,500]
num_spam=[400,800,1200,1600,2000,2400,2800,3200,3600,4000]
def linear_svc():
	pipe=Pipeline([('scaler',StandardScaler()),('svm',SVC(kernel='linear'))])
	train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
	acc_train_cancer_list=[]
	acc_test_cancer_list=[]
	for num in num_cancer:
		pipe.fit(train_features_cancer[:num,:],train_labels_cancer[:num])
		acc_train_cancer=pipe.score(train_features_cancer[:num,:],train_labels_cancer[:num])
		acc_test_cancer=pipe.score(test_features_cancer,test_labels_cancer)
		acc_train_cancer_list.append(acc_train_cancer)
		acc_test_cancer_list.append(acc_test_cancer)

	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	acc_train_spam_list=[]
	acc_test_spam_list=[]
	for num in num_spam:
		pipe.fit(train_features_spam[:num,:],train_labels_spam[:num])
		acc_train_spam=pipe.score(train_features_spam[:num,:],train_labels_spam[:num])
		acc_test_spam=pipe.score(test_features_spam,test_labels_spam)
		acc_train_spam_list.append(acc_train_spam)
		acc_test_spam_list.append(acc_test_spam)
	plt.figure(figsize=(10,6))
	plt.subplot(121)
	plt.plot(num_cancer,acc_train_cancer_list,label='train')
	plt.plot(num_cancer,acc_test_cancer_list,label='test')
	plt.xlabel('training size')
	plt.ylabel('accuracy')
	plt.title('linear cancer smv performance \nvs training size')
	plt.legend(loc='upper right')
	plt.subplot(122)
	plt.plot(num_spam,acc_train_spam_list,label='train')
	plt.plot(num_spam,acc_test_spam_list,label='test')
	plt.xlabel('training size')
	plt.ylabel('accuracy')
	plt.title('linear spam smv performance \nvs training size')
	plt.legend(loc='upper right')
	plt.show()


def poly_svc():
	pipe=Pipeline([('scaler',StandardScaler()),('svm',SVC(kernel='poly',degree=2))])
	train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
	acc_train_cancer_list=[]
	acc_test_cancer_list=[]
	for num in num_cancer:
		pipe.fit(train_features_cancer[:num,:],train_labels_cancer[:num])
		acc_train_cancer=pipe.score(train_features_cancer[:num,:],train_labels_cancer[:num])
		acc_test_cancer=pipe.score(test_features_cancer,test_labels_cancer)
		acc_train_cancer_list.append(acc_train_cancer)
		acc_test_cancer_list.append(acc_test_cancer)

	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	acc_train_spam_list=[]
	acc_test_spam_list=[]
	for num in num_spam:
		pipe.fit(train_features_spam[:num,:],train_labels_spam[:num])
		acc_train_spam=pipe.score(train_features_spam[:num,:],train_labels_spam[:num])
		acc_test_spam=pipe.score(test_features_spam,test_labels_spam)
		acc_train_spam_list.append(acc_train_spam)
		acc_test_spam_list.append(acc_test_spam)
	plt.figure(figsize=(10,6))
	plt.subplot(121)
	plt.plot(num_cancer,acc_train_cancer_list,label='train')
	plt.plot(num_cancer,acc_test_cancer_list,label='test')
	plt.xlabel('training size')
	plt.ylabel('accuracy')
	plt.title('poly kernel cancer smv performance \nvs training size')
	plt.legend(loc='upper right')
	plt.subplot(122)
	plt.plot(num_spam,acc_train_spam_list,label='train')
	plt.plot(num_spam,acc_test_spam_list,label='test')
	plt.xlabel('training size')
	plt.ylabel('accuracy')
	plt.title('poly kernel spam smv performance \nvs training size')
	plt.legend(loc='upper right')
	plt.show()

def rbf_svc():
	pipe=Pipeline([('scaler',StandardScaler()),('svm',SVC(kernel='rbf',gamma=5.e-1))])
	train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
	acc_train_cancer_list=[]
	acc_test_cancer_list=[]
	for num in num_cancer:
		pipe.fit(train_features_cancer[:num,:],train_labels_cancer[:num])
		acc_train_cancer=pipe.score(train_features_cancer[:num,:],train_labels_cancer[:num])
		acc_test_cancer=pipe.score(test_features_cancer,test_labels_cancer)
		acc_train_cancer_list.append(acc_train_cancer)
		acc_test_cancer_list.append(acc_test_cancer)

	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	acc_train_spam_list=[]
	acc_test_spam_list=[]
	pipe.set_params(svm__gamma=5.e-2)
	for num in num_spam:
		pipe.fit(train_features_spam[:num,:],train_labels_spam[:num])
		acc_train_spam=pipe.score(train_features_spam[:num,:],train_labels_spam[:num])
		acc_test_spam=pipe.score(test_features_spam,test_labels_spam)
		acc_train_spam_list.append(acc_train_spam)
		acc_test_spam_list.append(acc_test_spam)
	plt.figure(figsize=(10,6))
	plt.subplot(121)
	plt.plot(num_cancer,acc_train_cancer_list,label='train')
	plt.plot(num_cancer,acc_test_cancer_list,label='test')
	plt.xlabel('training size')
	plt.ylabel('accuracy')
	plt.title('rbf kernel cancer smv performance \nvs training size')
	plt.legend(loc='upper right')
	plt.subplot(122)
	plt.plot(num_spam,acc_train_spam_list,label='train')
	plt.plot(num_spam,acc_test_spam_list,label='test')
	plt.xlabel('training size')
	plt.ylabel('accuracy')
	plt.title('rbf kernel spam smv performance \nvs training size')
	plt.legend(loc='upper right')
	plt.show()


def performance_vs_gamma():
	gamma_list=[2**(-10),2**(-9),2**(-8),2**(-7),2**(-6),2**(-5),2**(-4),2**(-3),2**(-2),2**(-1),2**0]
	pipe=Pipeline([('scaler',StandardScaler()),('svm',SVC(kernel='rbf'))])
	acc_train_cancer_list=[]
	acc_test_cancer_list=[]
	train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
	for gamma in gamma_list:
		pipe.set_params(svm__gamma=gamma)
		pipe.fit(train_features_cancer,train_labels_cancer)
		acc_train_cancer=pipe.score(train_features_cancer,train_labels_cancer)
		acc_test_cancer=pipe.score(test_features_cancer,test_labels_cancer)
		acc_train_cancer_list.append(acc_train_cancer)
		acc_test_cancer_list.append(acc_test_cancer)

	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	acc_train_spam_list=[]
	acc_test_spam_list=[]
	for gamma in gamma_list:
		pipe.set_params(svm__gamma=gamma)
		pipe.fit(train_features_spam,train_labels_spam)
		acc_train_spam=pipe.score(train_features_spam,train_labels_spam)
		acc_test_spam=pipe.score(test_features_spam,test_labels_spam)
		acc_train_spam_list.append(acc_train_spam)
		acc_test_spam_list.append(acc_test_spam)
	plt.figure(figsize=(10,6))
	plt.subplot(121)
	plt.plot(gamma_list,acc_train_cancer_list,label='train')
	plt.plot(gamma_list,acc_test_cancer_list,label='test')
	plt.xscale('log')
	plt.xlabel('gamma')
	plt.ylabel('accuracy')
	plt.title('rbf kernel cancer classifier \nperformance vs gamma')
	plt.subplot(122)
	plt.plot(gamma_list,acc_train_spam_list,label='train')
	plt.plot(gamma_list,acc_test_spam_list,label='test')
	plt.xscale('log')
	plt.xlabel('gamma')
	plt.ylabel('accuracy')
	plt.title('rbf kernel spam classifier \nperformance vs gamma')
	plt.show()

def gridSearch():
	pipe=Pipeline([('scaler',StandardScaler()),('svm',SVC())])
	params=[{'svm__kernel':['linear']},{'svm__kernel':['poly'],'svm__degree':[2,3,4,5,6,7,8,9,10]},{'svm__kernel':['rbf'],'svm__gamma':[2**(-10),2**(-9),2**(-8),2**(-7),2**(-6),2**(-5),2**(-4),2**(-3),2**(-2),2**(-1),2**0]}]
	clf=GridSearchCV(pipe,param_grid=params,cv=5)
	train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
	clf.fit(train_features_cancer,train_labels_cancer)
	best_train_acc_cancer=clf.score(train_features_cancer,train_labels_cancer)
	best_test_acc_cancer=clf.score(test_features_cancer,test_labels_cancer)
	print('best train accuracy for cancer SVM is:',best_train_acc_cancer)
	print('best test accuracy for cancer SVM is:',best_test_acc_cancer)
	print('best parameters for cancer SVM is:',clf.best_params_)

	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	clf.fit(train_features_spam,train_labels_spam)
	best_train_acc_spam=clf.score(train_features_spam,train_labels_spam)
	best_test_acc_spam=clf.score(test_features_spam,test_labels_spam)
	print('best train accuracy for spam SVM is:',best_train_acc_spam)
	print('best test accuracy for spam SVM is:',best_test_acc_spam)
	print('best parameters for spam SVM is:',clf.best_params_)




if __name__=="__main__":
	linear_svc()
	poly_svc()
	rbf_svc()
	performance_vs_gamma()
	gridSearch()



