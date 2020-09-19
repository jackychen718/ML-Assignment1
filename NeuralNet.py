from assignment1 import split_train_test_spam,split_train_test_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

num_cancer=[50,100,150,200,250,300,350,400,450,500]
num_spam=[400,800,1200,1600,2000,2400,2800,3200,3600,4000]
def zero_hidden_model(X_train,y_train,X_test,y_test,iter_num=2000):
    tf.reset_default_graph()
    n_feature=X_train.shape[1]
    X=tf.placeholder(tf.float32,shape=(None,n_feature))
    Y=tf.placeholder(tf.float32,shape=(None))
    w1=tf.get_variable(name='w1',shape=(n_feature,1),dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform())
    b1=tf.get_variable(name='b1',shape=(1,1),dtype=tf.float32,initializer=tf.zeros_initializer())
    z=tf.reshape(tf.add(tf.matmul(X,w1),b1),[-1])
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=z))
    opt=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    init=tf.global_variables_initializer()
    train_acc=0.
    test_acc=0.
    iter_list=[]
    cost_list=[]
    with tf.Session() as sess:
        sess.run(init)
        for iter_i in range(iter_num):
            _,cost=sess.run([opt,loss],feed_dict={X:X_train,Y:y_train})
            if iter_i%10==0:
                iter_list.append(iter_i)
                cost_list.append(cost)
        train_predict=np.array(sess.run(z,feed_dict={X:X_train})>0,dtype=int)
        test_predict=np.array(sess.run(z,feed_dict={X:X_test})>0,dtype=int)
        train_acc=np.sum(train_predict==y_train)/len(train_predict)
        test_acc=np.sum(test_predict==y_test)/len(test_predict)
    return (iter_list,cost_list),(train_acc,test_acc)
    
def  one_hidden_model(X_train,y_train,X_test,y_test,iter_num=2000,hidden_units=16):
    tf.reset_default_graph()
    n_feature=X_train.shape[1]
    X=tf.placeholder(tf.float32,shape=(None,n_feature))
    Y=tf.placeholder(tf.float32,shape=(None))
    w1=tf.get_variable(name='w1',shape=(n_feature,hidden_units),dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform())
    b1=tf.get_variable(name='b1',shape=(1,hidden_units),dtype=tf.float32,initializer=tf.zeros_initializer())
    w2=tf.get_variable(name='w2',shape=(hidden_units,1),dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform())
    b2=tf.get_variable(name='b2',shape=(1,1),dtype=tf.float32,initializer=tf.zeros_initializer())
    z1=tf.add(tf.matmul(X,w1),b1)
    a1=tf.nn.relu(z1)
    z=tf.reshape(tf.add(tf.matmul(a1,w2),b2),[-1])
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=z))
    opt=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    init=tf.global_variables_initializer()
    train_acc=0.
    test_acc=0.
    iter_list=[]
    cost_list=[]
    with tf.Session() as sess:
        sess.run(init)
        for iter_i in range(iter_num):
            _,cost=sess.run([opt,loss],feed_dict={X:X_train,Y:y_train})
            if iter_i%10==0:
                iter_list.append(iter_i)
                cost_list.append(cost)
        train_predict=np.array(sess.run(z,feed_dict={X:X_train})>0,dtype=int)
        test_predict=np.array(sess.run(z,feed_dict={X:X_test})>0,dtype=int)
        train_acc=np.sum(train_predict==y_train)/len(train_predict)
        test_acc=np.sum(test_predict==y_test)/len(test_predict)
    return (iter_list,cost_list),(train_acc,test_acc)
        
def converge_time():
    scaler=StandardScaler()
    train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
    scaler.fit(train_features_cancer)
    train_features_cancer_norm=scaler.transform(train_features_cancer)
    test_features_cancer_norm=scaler.transform(test_features_cancer)
    cost_vs_iter_cancer,acc_cancer=zero_hidden_model(train_features_cancer_norm,train_labels_cancer,test_features_cancer_norm,test_labels_cancer)
    iter_list_cancer,cost_list_cancer=cost_vs_iter_cancer
    acc_train_cancer,acc_test_cancer=acc_cancer
    print('training accuracy for breast cancer learner is:',acc_train_cancer)
    print('test accuracy for breast cancer learner is:',acc_test_cancer)
    
    train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
    scaler.fit(train_features_spam)
    train_features_spam_norm=scaler.transform(train_features_spam)
    test_features_spam_norm=scaler.transform(test_features_spam)
    cost_vs_iter_spam,acc_spam=zero_hidden_model(train_features_spam_norm,train_labels_spam,test_features_spam_norm,test_labels_spam)
    iter_list_spam,cost_list_spam=cost_vs_iter_spam
    acc_train_spam,acc_test_spam=acc_spam
    print('training accuracy for spam learner is:',acc_train_spam)
    print('test accuracy for spam learner is:',acc_test_spam)
    
    plt.figure(figsize=(8,3.5))
    plt.subplot(121)
    plt.plot(iter_list_cancer,cost_list_cancer)
    plt.title('cost_vs_iter for breast cancer')
    plt.xlabel('iter_num')
    plt.ylabel('entropy loss')
    plt.subplot(122)
    plt.plot(iter_list_spam,cost_list_spam)
    plt.title('cost_vs_iter for spam')
    plt.xlabel('iter_num')
    plt.ylabel('entropy loss')
    
def performance_curve_zero_hidden():
    scaler=StandardScaler()
    train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
    acc_train_list_cancer=[]
    acc_test_list_cancer=[]
    for num in num_cancer:
        scaler.fit(train_features_cancer[:num,:])
        train_features_norm_cancer=scaler.transform(train_features_cancer[:num,:])
        test_features_norm_cancer=scaler.transform(test_features_cancer)
        _,acc_cancer=zero_hidden_model(train_features_norm_cancer,train_labels_cancer[:num],test_features_norm_cancer,test_labels_cancer)
        acc_train_cancer,acc_test_cancer=acc_cancer
        acc_train_list_cancer.append(acc_train_cancer)
        acc_test_list_cancer.append(acc_test_cancer)
        
    train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
    acc_train_list_spam=[]
    acc_test_list_spam=[]
    for num in num_spam:
        scaler.fit(train_features_spam[:num,:])
        train_features_norm_spam=scaler.transform(train_features_spam[:num,:])
        test_features_norm_spam=scaler.transform(test_features_spam)
        _,acc_spam=zero_hidden_model(train_features_norm_spam,train_labels_spam[:num],test_features_norm_spam,test_labels_spam)
        acc_train_spam,acc_test_spam=acc_spam
        acc_train_list_spam.append(acc_train_spam)
        acc_test_list_spam.append(acc_test_spam)
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(num_cancer,acc_train_list_cancer,label='train acc')
    plt.plot(num_cancer,acc_test_list_cancer,label='test acc')
    plt.xlabel('training size')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.title('accuracy vs trainig size for cancer classifier \nwith no hidden layer')
    plt.subplot(122)
    plt.plot(num_spam,acc_train_list_spam,label='train acc')
    plt.plot(num_spam,acc_test_list_spam,label='test acc')
    plt.xlabel('training size')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.title('accuracy vs trainig size for spam classifier \nwith no hidden layer')
    plt.show()

def performance_curve_one_hidden():
    scaler=StandardScaler()
    train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
    acc_train_list_cancer=[]
    acc_test_list_cancer=[]
    for num in num_cancer:
        scaler.fit(train_features_cancer[:num,:])
        train_features_norm_cancer=scaler.transform(train_features_cancer[:num,:])
        test_features_norm_cancer=scaler.transform(test_features_cancer)
        _,acc_cancer=one_hidden_model(train_features_norm_cancer,train_labels_cancer[:num],test_features_norm_cancer,test_labels_cancer)
        acc_train_cancer,acc_test_cancer=acc_cancer
        acc_train_list_cancer.append(acc_train_cancer)
        acc_test_list_cancer.append(acc_test_cancer)
        
    train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
    acc_train_list_spam=[]
    acc_test_list_spam=[]
    for num in num_spam:
        scaler.fit(train_features_spam[:num,:])
        train_features_norm_spam=scaler.transform(train_features_spam[:num,:])
        test_features_norm_spam=scaler.transform(test_features_spam)
        _,acc_spam=one_hidden_model(train_features_norm_spam,train_labels_spam[:num],test_features_norm_spam,test_labels_spam)
        acc_train_spam,acc_test_spam=acc_spam
        acc_train_list_spam.append(acc_train_spam)
        acc_test_list_spam.append(acc_test_spam)
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(num_cancer,acc_train_list_cancer,label='train acc')
    plt.plot(num_cancer,acc_test_list_cancer,label='test acc')
    plt.xlabel('training size')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.title('accuracy vs trainig size for cancer classifier \nwith one hidden layer 16 hidden units')
    plt.subplot(122)
    plt.plot(num_spam,acc_train_list_spam,label='train acc')
    plt.plot(num_spam,acc_test_list_spam,label='test acc')
    plt.xlabel('training size')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.title('accuracy vs trainig size for spam classifier \nwith one hidden layer 16 hidden units')
    plt.show()    


def performance_curve_one_hidden_few_units():
    scaler=StandardScaler()
    train_features_cancer,train_labels_cancer,test_features_cancer,test_labels_cancer=split_train_test_breast_cancer()
    acc_train_list_cancer=[]
    acc_test_list_cancer=[]
    for num in num_cancer:
        scaler.fit(train_features_cancer[:num,:])
        train_features_norm_cancer=scaler.transform(train_features_cancer[:num,:])
        test_features_norm_cancer=scaler.transform(test_features_cancer)
        _,acc_cancer=one_hidden_model(train_features_norm_cancer,train_labels_cancer[:num],test_features_norm_cancer,test_labels_cancer,hidden_units=4)
        acc_train_cancer,acc_test_cancer=acc_cancer
        acc_train_list_cancer.append(acc_train_cancer)
        acc_test_list_cancer.append(acc_test_cancer)
        
    train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
    acc_train_list_spam=[]
    acc_test_list_spam=[]
    for num in num_spam:
        scaler.fit(train_features_spam[:num,:])
        train_features_norm_spam=scaler.transform(train_features_spam[:num,:])
        test_features_norm_spam=scaler.transform(test_features_spam)
        _,acc_spam=one_hidden_model(train_features_norm_spam,train_labels_spam[:num],test_features_norm_spam,test_labels_spam,hidden_units=4)
        acc_train_spam,acc_test_spam=acc_spam
        acc_train_list_spam.append(acc_train_spam)
        acc_test_list_spam.append(acc_test_spam)
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(num_cancer,acc_train_list_cancer,label='train acc')
    plt.plot(num_cancer,acc_test_list_cancer,label='test acc')
    plt.xlabel('training size')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.title('accuracy vs trainig size for cancer classifier \nwith one hidden layer 4 hidden units')
    plt.subplot(122)
    plt.plot(num_spam,acc_train_list_spam,label='train acc')
    plt.plot(num_spam,acc_test_list_spam,label='test acc')
    plt.xlabel('training size')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.title('accuracy vs trainig size for spam classifier \nwith one hidden layer 4 hidden units')
    plt.show()    

def  best_spam_classifier_stocastic(X_train,y_train,X_test,y_test,iter_num=2000,hidden_units=4):
    tf.reset_default_graph()
    train_batches=[]
    for i in range(100):
        X_train_mini_batch=X_train[i*40:(i+1)*40,:]
        y_train_mini_batch=y_train[i*40:(i+1)*40]
        train_batches.append((X_train_mini_batch,y_train_mini_batch))
    n_feature=X_train.shape[1]
    X=tf.placeholder(tf.float32,shape=(None,n_feature))
    Y=tf.placeholder(tf.float32,shape=(None))
    w1=tf.get_variable(name='w1',shape=(n_feature,hidden_units),dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform())
    b1=tf.get_variable(name='b1',shape=(1,hidden_units),dtype=tf.float32,initializer=tf.zeros_initializer())
    w2=tf.get_variable(name='w2',shape=(hidden_units,1),dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform())
    b2=tf.get_variable(name='b2',shape=(1,1),dtype=tf.float32,initializer=tf.zeros_initializer())
    z1=tf.add(tf.matmul(X,w1),b1)
    a1=tf.nn.relu(z1)
    z=tf.reshape(tf.add(tf.matmul(a1,w2),b2),[-1])
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=z))
    opt=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    init=tf.global_variables_initializer()
    iter_list=[]
    cost_list=[]
    with tf.Session() as sess:
        sess.run(init)
        for iter_i in range(iter_num):
            aver_cost=0.
            for each_mini_batch in train_batches:
                X_train_mini_batch,y_train_mini_batch=each_mini_batch
                _,cost=sess.run([opt,loss],feed_dict={X:X_train_mini_batch,Y:y_train_mini_batch})
                aver_cost+=cost/100.
            if iter_i%10==0:
                iter_list.append(iter_i)
                cost_list.append(aver_cost)
    plt.plot(iter_list,cost_list)
    plt.xlabel('epoch num')
    plt.ylabel('loss')
    plt.title('stocastic descent for spam classifier')
    plt.show()


def  best_spam_classifier_batch(X_train,y_train,X_test,y_test,iter_num=2000,hidden_units=4):
    tf.reset_default_graph()
    n_feature=X_train.shape[1]
    X=tf.placeholder(tf.float32,shape=(None,n_feature))
    Y=tf.placeholder(tf.float32,shape=(None))
    w1=tf.get_variable(name='w1',shape=(n_feature,hidden_units),dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform())
    b1=tf.get_variable(name='b1',shape=(1,hidden_units),dtype=tf.float32,initializer=tf.zeros_initializer())
    w2=tf.get_variable(name='w2',shape=(hidden_units,1),dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform())
    b2=tf.get_variable(name='b2',shape=(1,1),dtype=tf.float32,initializer=tf.zeros_initializer())
    z1=tf.add(tf.matmul(X,w1),b1)
    a1=tf.nn.relu(z1)
    z=tf.reshape(tf.add(tf.matmul(a1,w2),b2),[-1])
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=z))
    opt=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    init=tf.global_variables_initializer()
    iter_list=[]
    cost_list=[]
    with tf.Session() as sess:
        sess.run(init)
        for iter_i in range(iter_num):
            _,cost=sess.run([opt,loss],feed_dict={X:X_train,Y:y_train})
            if iter_i%10==0:
                iter_list.append(iter_i)
                cost_list.append(cost)
    plt.plot(iter_list,cost_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('batch descent for spam classifier')
    plt.show()
    
def batch_vs_stocastic():
    scaler=StandardScaler()
    train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
    scaler.fit(train_features_spam)
    train_features_spam_norm=scaler.transform(train_features_spam)
    test_features_spam_norm=scaler.transform(test_features_spam)
    best_spam_classifier_batch(train_features_spam_norm,train_labels_spam,test_features_spam_norm,test_labels_spam)
    best_spam_classifier_stocastic(train_features_spam_norm,train_labels_spam,test_features_spam_norm,test_labels_spam)

if __name__=="__main__":
    converge_time()
    performance_curve_zero_hidden()
    performance_curve_one_hidden()
    performance_curve_one_hidden_few_units()
    batch_vs_stocastic()