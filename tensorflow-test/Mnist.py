import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

#input 一张图片784维，数量不定
x = tf.placeholder("float", shape=[None, 784])
#output 每一行是一个10维的one-hot量，代表对应的某一mnist图片的类别
y_ = tf.placeholder("float", shape=[None, 10])

#Weight_initial=0
w = tf.Variable(tf.zeros([784, 10]))
#bias_initial=0
b = tf.Variable(tf.zeros([10]))
#Assign the initial values to each variable
sess.run(tf.global_variables_initializer())

#softmax
y = tf.nn.softmax(tf.matmul(x, w) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#train
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#evaluate the result
#argmax:给出tensor对象在某一维上其数据最大值所在的索引值
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
