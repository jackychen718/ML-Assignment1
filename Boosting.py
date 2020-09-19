from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from assignment1 import split_train_test_breast_cancer,split_train_test_spam

num_spam=[400,800,1200,1600,2000,2400,2800,3200,3600,4000]
def accuracy_vs_num_tree():
	max_depth_tree=DecisionTreeClassifier(max_depth=3)
	num_trees_list=[i+1 for i in range(100)]
	train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
	acc_train_cancer_list=[]
	acc_test_cancer_list=[]
	boost_classifier=AdaBoostClassifier(max_depth_tree,n_estimators=1)
	for num_trees in num_trees_list:
		boost_classifier.set_params(n_estimators=num_trees)
		boost_classifier.fit(train_features_cancer,train_labels_cancer)
		acc_train_cancer=boost_classifier.score(train_features_cancer,train_labels_cancer)
		acc_train_cancer_list.append(acc_train_cancer)
		acc_test_cancer=boost_classifier.score(test_features_cancer,test_labels_cancer)
		acc_test_cancer_list.append(acc_test_cancer)

	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	acc_train_spam_list=[]
	acc_test_spam_list=[]
	for num_trees in num_trees_list:
		boost_classifier.set_params(n_estimators=num_trees)
		boost_classifier.fit(train_features_spam,train_labels_spam)
		acc_train_spam=boost_classifier.score(train_features_spam,train_labels_spam)
		acc_train_spam_list.append(acc_train_spam)
		acc_test_spam=boost_classifier.score(test_features_spam,test_labels_spam)
		acc_test_spam_list.append(acc_test_spam)
	plt.figure(figsize=(10,6))
	plt.subplot(121)
	plt.plot(num_trees_list,acc_train_cancer_list,label='train')
	plt.plot(num_trees_list,acc_test_cancer_list,label='test')
	plt.xlabel('num of trees')
	plt.ylabel('accuracy')
	plt.title('cancer accuracy vs number of boosting trees')
	plt.legend(loc='upper right')
	plt.subplot(122)
	plt.plot(num_trees_list,acc_train_spam_list,label='train')
	plt.plot(num_trees_list,acc_test_spam_list,label='test')
	plt.xlabel('num of trees')
	plt.ylabel('accuracy')
	plt.title('spam accuracy vs number of boosting trees')
	plt.legend(loc='upper right')
	plt.show()

def post_pruning_boosting_tree_performance():
	pruning_tree=DecisionTreeClassifier(ccp_alpha=0.015)
	num_trees_list=[i+1 for i in range(20)]
	train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
	acc_train_cancer_list=[]
	acc_test_cancer_list=[]
	boost_classifier=AdaBoostClassifier(pruning_tree,n_estimators=1)
	for num_trees in num_trees_list:
		boost_classifier.set_params(n_estimators=num_trees)
		boost_classifier.fit(train_features_cancer,train_labels_cancer)
		acc_train_cancer=boost_classifier.score(train_features_cancer,train_labels_cancer)
		acc_train_cancer_list.append(acc_train_cancer)
		acc_test_cancer=boost_classifier.score(test_features_cancer,test_labels_cancer)
		acc_test_cancer_list.append(acc_test_cancer)

	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	acc_train_spam_list=[]
	acc_test_spam_list=[]
	for num_trees in num_trees_list:
		boost_classifier.set_params(base_estimator__ccp_alpha=0.005,n_estimators=num_trees)
		boost_classifier.fit(train_features_spam,train_labels_spam)
		acc_train_spam=boost_classifier.score(train_features_spam,train_labels_spam)
		acc_train_spam_list.append(acc_train_spam)
		acc_test_spam=boost_classifier.score(test_features_spam,test_labels_spam)
		acc_test_spam_list.append(acc_test_spam)
	plt.figure(figsize=(10,6))
	plt.subplot(121)
	plt.plot(num_trees_list,acc_train_cancer_list,label='train')
	plt.plot(num_trees_list,acc_test_cancer_list,label='test')
	plt.xlabel('num of trees')
	plt.ylabel('accuracy')
	plt.title('post-pruning boosting cancer classifer \nperformance vs number of boosting trees')
	plt.legend(loc='upper right')
	plt.subplot(122)
	plt.plot(num_trees_list,acc_train_spam_list,label='train')
	plt.plot(num_trees_list,acc_test_spam_list,label='test')
	plt.xlabel('num of trees')
	plt.ylabel('accuracy')
	plt.title('post-pruning boosting spam classifer \nperformance vs number of boosting trees')
	plt.legend(loc='upper right')
	plt.show()

def spam_classifier_with_different_training_size():
	max_depth_tree=DecisionTreeClassifier(max_depth=3)
	clf=AdaBoostClassifier(max_depth_tree,n_estimators=10)
	train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
	acc_train_list=[]
	acc_test_list=[]
	for num in num_spam:
		clf.fit(train_features_spam[:num,:],train_labels_spam[:num])
		acc_train=clf.score(train_features_spam[:num,:],train_labels_spam[:num])
		acc_test=clf.score(test_features_spam,test_labels_spam)
		acc_train_list.append(acc_train)
		acc_test_list.append(acc_test)

	pruning_tree=DecisionTreeClassifier(ccp_alpha=0.012)
	clf_pruning=AdaBoostClassifier(pruning_tree,n_estimators=10)
	acc_train_list_post=[]
	acc_test_list_post=[]
	for num in num_spam:
		clf_pruning.fit(train_features_spam[:num,:],train_labels_spam[:num])
		acc_train_post=clf_pruning.score(train_features_spam[:num,:],train_labels_spam[:num])
		acc_test_post=clf_pruning.score(test_features_spam,test_labels_spam)
		acc_train_list_post.append(acc_train_post)
		acc_test_list_post.append(acc_test_post)

	plt.figure(figsize=(10,6))
	plt.subplot(121)
	plt.plot(num_spam,acc_train_list,label='train')
	plt.plot(num_spam,acc_test_list,label='test')
	plt.xlabel('training size')
	plt.ylabel('accuracy')
	plt.title('boosting spam classifier with pre-pruning trees \nperformance vs training size')
	plt.subplot(122)
	plt.plot(num_spam,acc_train_list_post,label='train')
	plt.plot(num_spam,acc_test_list_post,label='test')
	plt.xlabel('training size')
	plt.ylabel('accuracy')
	plt.title('boosting spam classifier with post-pruning trees \nperformance vs training size')
	plt.show()

if __name__=="__main__":
	accuracy_vs_num_tree()
	post_pruning_boosting_tree_performance()
	spam_classifier_with_different_training_size()
