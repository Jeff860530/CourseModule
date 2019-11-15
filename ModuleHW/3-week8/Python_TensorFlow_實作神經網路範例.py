import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
###I. 用TensorFlow計算Linear Regression (梯度法)
X = np.linspace(-1, 1, 100)[:,np.newaxis]
np.random.shuffle(X)    # randomize the data
Y =  3 * X + 2 + np.random.normal(0, 0.05, (100, 1)) # 線性

X_train, Y_train = X[:70], Y[:70]     # train 前 60 data points
X_test, Y_test = X[70:], Y[70:]      

dim_input=1;
dim_output=1;
###########  Linear Regression by TensorFlow (GradientDescent) ###############
xs = tf.placeholder(tf.float32,[None,dim_input],name='x_input')
ys = tf.placeholder(tf.float32,[None,dim_output],name='y_output')

w = tf.Variable(tf.random_normal([dim_input, dim_output]))
b = tf.Variable(tf.zeros([1, dim_output]) + 1,)
prediction = tf.matmul(xs, w) + b
# loss function (prediction and real target)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-ys),
     reduction_indices=[1]),name='Loss')
# optimal: find the Weights and biases
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


rmse=np.array([])
for i in range(1000):
    sess.run(train_step,feed_dict={xs: X_train, ys: Y_train})
    y_pre = sess.run(prediction, feed_dict={xs: X_test})
    rmse=np.append(rmse,np.mean(np.sqrt(np.sum(np.square(Y_test-y_pre)))))
    if i>=1:
        if abs(rmse[i]-rmse[i-1])<np.spacing(1):
            break
    print('Iternation' + str(i+1) + ': RMSE=' + str(rmse[i]))
    
#####II. 用Closedform計算Linear Regression (唯一解)
def Train_LSE_Regression(x,y):
      tmp1=np.matmul(np.transpose(x),x)
      tmp1=np.linalg.pinv(tmp1)
      tmp2=np.matmul(np.transpose(y),x)
      Weight=np.matmul(tmp1,np.transpose(tmp2))
      return Weight

n_train=np.size(X_train,0)
n_test=np.size(X_test,0)
X_input_train=np.concatenate((X_train,np.ones([n_train,1])),axis=1)
Weight_LSE=Train_LSE_Regression(X_input_train,Y_train)

X_input_test=np.concatenate((X_test,np.ones([n_test,1])),axis=1)
y_pre_LSE=np.matmul(X_input_test,Weight_LSE)
mse_LSE=np.mean(np.sqrt(np.sum(np.square(Y_test-y_pre))))
  
  
  
print('TF (GD) Weights:' + str(sess.run(w)) + str(sess.run(b)))
print('TF (GD) MSE:' + str(mse_tf))
print('LSE     Weights:' + str(Weight_LSE[0]) + str(Weight_LSE[1]))
print('LSE     MSE:' + str(mse_LSE))