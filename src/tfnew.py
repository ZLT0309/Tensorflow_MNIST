import tensorflow as tf
import numpy as np
from matplotlib import  pyplot as plt

'''
a = tf.constant(3,dtype=tf.float32)
b = tf.constant(3,dtype=np.float32)

print("a=",a)
print("value a =", a.numpy())
print("b=",b)

random_float = tf.random.uniform(shape=(2,3,7),dtype=tf.float32)
print("random_float=", random_float)

zero_vector=tf.zeros(shape=(2,3),dtype=tf.float32)
print("zero_vector=", zero_vector)

A = tf.constant([[1.0,2.0],[3.0,4.0]],dtype=tf.float32)
B = tf.constant([[5.0,6.0],[7.0,8.0]],dtype=tf.float32)
print(A.numpy())

C = tf.add(A, B)
D = tf.matmul(A, B)

print(D.numpy())

x = tf.Variable(initial_value=3.0)
with tf.GradientTape() as tape:
    y = tf.square(x)
    # y = x * x
y_grad = tape.gradient(y,x)
print([y,y_grad])
'''
'''
X = tf.constant([[1.0,2.0],[3.0,4.0]])
y = tf.constant([[1.0],[2.0]])

w = tf.Variable(initial_value=[[1.0 ],[2.0]])
b = tf.Variable(initial_value=1.0)

with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X,w) + b - y))
w_grad,b_grad = tape.gradient(L,[w,b])
print(L.numpy(),w_grad.numpy(),b_grad.numpy())
'''



'''
learning_rate = tf.constant(0.001)

x = tf.Variable(initial_value=10,dtype=tf.float32)

for e in range(100):
    with tf.GradientTape() as tape:
        y = x * x - 6 * x + 9
        loss = tf.square(y-0)

    y_grad = tape.gradient(loss,x)

    x.assign_sub(learning_rate * y_grad)
print(x.numpy())
'''

'''
lr = tf.constant(0.001)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


x = tf.Variable(initial_value=10,dtype=tf.float32)

for e in range(10000):
    with tf.GradientTape() as tape:
        y = x * x - 6 * x + 9
        loss = tf.square(y-0)

    y_grad = tape.gradient(loss,x)

    print(loss.numpy(),x.numpy())
    optimizer.apply_gradients(grads_and_vars=zip([y_grad],[x]))
    print(x.numpy())
'''

#linear aggresive  by numpy
'''
X_raw = np.array([201.3,2014,2015,2016,2017],dtype=np.float32)
y_raw = np.array([12000,14000,15000,16600,17500],dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

a,b = 0,0

num_epoch = 10000
learning_rate = 1e-3

for e in range(num_epoch):
    y_pred = a * X + b
    grad_a,grad_b = (y_pred-y).dot(X),(y_pred-y).sum()

    a,b = a- learning_rate * grad_a,b-learning_rate*grad_b

print(a,b)

'''

X_raw = np.array([201.3,2014,2015,2016,2017],dtype=np.float32)
y_raw = np.array([12000,14000,15000,16600,17500],dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

X = tf.constant(X,dtype=tf.float32)
y = tf.constant(y,dtype=tf.float32)

a = tf.Variable(initial_value=0,dtype=tf.float32)
b = tf.Variable(initial_value=0,dtype=tf.float32)
variables = [a,b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
for e in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred-y))

    grads = tape.gradient(loss,variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads,variables))

print(a.numpy(),b.numpy())