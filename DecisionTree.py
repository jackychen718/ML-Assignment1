from assignment1 import split_train_test_spam,split_train_test_breast_cancer
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


num_cancer=[50,100,150,200,250,300,350,400,450,500]
num_spam=[400,800,1200,1600,2000,2400,2800,3200,3600,4000]

def full_tree():
    accuracy_cancer_train=[]
    accuracy_cancer_test=[]
    accuracy_spam_train=[]
    accuracy_spam_test=[]
    clf=tree.DecisionTreeClassifier()

    train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
    for num in num_cancer:
    	clf.fit(train_features_cancer[:num,:],train_labels_cancer[:num])
    	acc_train_cancer=clf.score(train_features_cancer[:num,:],train_labels_cancer[:num])
    	accuracy_cancer_train.append(acc_train_cancer)
    	acc_test_cancer=clf.score(test_features_cancer,test_labels_cancer)
    	accuracy_cancer_test.append(acc_test_cancer)
    train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
    for num in num_spam:
    	clf.fit(train_features_spam[:num,:],train_labels_spam[:num])
    	acc_train_spam=clf.score(train_features_spam[:num,:],train_labels_spam[:num])
    	accuracy_spam_train.append(acc_train_spam)
    	acc_test_spam=clf.score(test_features_spam,test_labels_spam)
    	accuracy_spam_test.append(acc_test_spam)
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(num_cancer,accuracy_cancer_train,label='training')
    plt.plot(num_cancer,accuracy_cancer_test,label='test')
    plt.title('breast cancer train/test accuracy')
    plt.xlabel('size of training examples')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.subplot(122)
    plt.plot(num_spam,accuracy_spam_train,label='training')
    plt.plot(num_spam,accuracy_spam_test,label='test')
    plt.title('spam train/test accuracy')
    plt.xlabel('size of training examples')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.show()

def depth_limited_tree():
    accuracy_cancer_train=[]
    accuracy_cancer_test=[]
    accuracy_spam_train=[]
    accuracy_spam_test=[]
    clf=tree.DecisionTreeClassifier(max_depth=3)
    train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
    for num in num_cancer:
    	clf.fit(train_features_cancer[:num,:],train_labels_cancer[:num])
    	acc_train_cancer=clf.score(train_features_cancer[:num,:],train_labels_cancer[:num])
    	accuracy_cancer_train.append(acc_train_cancer)
    	acc_test_cancer=clf.score(test_features_cancer,test_labels_cancer)
    	accuracy_cancer_test.append(acc_test_cancer)
    train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
    clf.set_params(max_depth=3)
    for num in num_spam:
    	clf.fit(train_features_spam[:num,:],train_labels_spam[:num])
    	acc_train_spam=clf.score(train_features_spam[:num,:],train_labels_spam[:num])
    	accuracy_spam_train.append(acc_train_spam)
    	acc_test_spam=clf.score(test_features_spam,test_labels_spam)
    	accuracy_spam_test.append(acc_test_spam)
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(num_cancer,accuracy_cancer_train,label='training')
    plt.plot(num_cancer,accuracy_cancer_test,label='test')
    plt.title('breast cancer performance with depth of 3')
    plt.xlabel('size of training examples')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.subplot(122)
    plt.plot(num_spam,accuracy_spam_train,label='training')
    plt.plot(num_spam,accuracy_spam_test,label='test')
    plt.title('spam performance with depth of 3')
    plt.xlabel('size of training examples')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.show()

def leaf_num_limited_tree():
    accuracy_cancer_train=[]
    accuracy_cancer_test=[]
    accuracy_spam_train=[]
    accuracy_spam_test=[]
    clf=tree.DecisionTreeClassifier(min_samples_leaf=10)
    train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
    for num in num_cancer:
    	clf.fit(train_features_cancer[:num,:],train_labels_cancer[:num])
    	acc_train_cancer=clf.score(train_features_cancer[:num,:],train_labels_cancer[:num])
    	accuracy_cancer_train.append(acc_train_cancer)
    	acc_test_cancer=clf.score(test_features_cancer,test_labels_cancer)
    	accuracy_cancer_test.append(acc_test_cancer)
    train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
    clf.set_params(min_samples_leaf=10)
    for num in num_spam:
    	clf.fit(train_features_spam[:num,:],train_labels_spam[:num])
    	acc_train_spam=clf.score(train_features_spam[:num,:],train_labels_spam[:num])
    	accuracy_spam_train.append(acc_train_spam)
    	acc_test_spam=clf.score(test_features_spam,test_labels_spam)
    	accuracy_spam_test.append(acc_test_spam)
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(num_cancer,accuracy_cancer_train,label='training')
    plt.plot(num_cancer,accuracy_cancer_test,label='test')
    plt.title('breast cancer performance with \nat least 10 examples at leaf nodes')
    plt.xlabel('size of training examples')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.subplot(122)
    plt.plot(num_spam,accuracy_spam_train,label='training')
    plt.plot(num_spam,accuracy_spam_test,label='test')
    plt.title('spam performance with \nat least 10 examples at leaf nodes')
    plt.xlabel('size of training examples')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.show()

def post_pruning_tree():
    accuracy_cancer_train=[]
    accuracy_cancer_test=[]
    accuracy_spam_train=[]
    accuracy_spam_test=[]
    clf=tree.DecisionTreeClassifier(ccp_alpha=0.01)
    train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
    for num in num_cancer:
    	clf.fit(train_features_cancer[:num,:],train_labels_cancer[:num])
    	acc_train_cancer=clf.score(train_features_cancer[:num,:],train_labels_cancer[:num])
    	accuracy_cancer_train.append(acc_train_cancer)
    	acc_test_cancer=clf.score(test_features_cancer,test_labels_cancer)
    	accuracy_cancer_test.append(acc_test_cancer)
    train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
    clf.set_params(ccp_alpha=0.01)
    for num in num_spam:
    	clf.fit(train_features_spam[:num,:],train_labels_spam[:num])
    	acc_train_spam=clf.score(train_features_spam[:num,:],train_labels_spam[:num])
    	accuracy_spam_train.append(acc_train_spam)
    	acc_test_spam=clf.score(test_features_spam,test_labels_spam)
    	accuracy_spam_test.append(acc_test_spam)
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(num_cancer,accuracy_cancer_train,label='training')
    plt.plot(num_cancer,accuracy_cancer_test,label='test')
    plt.title('breast cancer performance with ccp_alpha 0.01')
    plt.xlabel('size of training examples')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.subplot(122)
    plt.plot(num_spam,accuracy_spam_train,label='training')
    plt.plot(num_spam,accuracy_spam_test,label='test')
    plt.title('spam performance with ccp_alpha 0.01')
    plt.xlabel('size of training examples')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.show()

def gridsearch():
	my_tree=tree.DecisionTreeClassifier()
	param_grid=[{'max_depth':[2,4,6,8,10]},{'min_samples_leaf':[2,4,8,16,32,64,128,256]},{'ccp_alpha':[0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128]}]
	clf=GridSearchCV(my_tree,param_grid,scoring='accuracy',cv=5)
	train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
	best_cancer_clf=clf.fit(train_features_cancer,train_labels_cancer)
	print('Best parameter for breast cancer classifier:',best_cancer_clf.best_params_)
	accuracy_cancer_train=best_cancer_clf.score(train_features_cancer,train_labels_cancer)
	accuracy_cancer_test=best_cancer_clf.score(test_features_cancer,test_labels_cancer)
	print('Training accuracy for best breast cancer classifier :',accuracy_cancer_train)
	print('Test accuracy for best breast cancer classifier :',accuracy_cancer_test)
	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	best_spam_clf=clf.fit(train_features_spam,train_labels_spam)
	accuracy_spam_train=best_spam_clf.score(train_features_spam,train_labels_spam)
	accuracy_spam_test=best_spam_clf.score(test_features_spam,test_labels_spam)
	print('--------------------------------------------------------------')
	print('Best parameter for spam classifier:',best_spam_clf.best_params_)
	print('Training accuracy for best spam classifier :',accuracy_spam_train)
	print('Test accuracy for best spam classifier :',accuracy_spam_test)


if __name__=="__main__":
	full_tree()
	depth_limited_tree()
	leaf_num_limited_tree()
	post_pruning_tree()
	gridsearch()

