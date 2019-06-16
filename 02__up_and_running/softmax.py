import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import warnings
warnings.simplefilter(action='ignore', category=Warning)



DATA_DIR = '/tmp/data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

# 1. 데이타를 내려받는다
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

# 2. input tensorflow를 구성한다
# input: batch, data vector size 28*28 = 784
x = tf.placeholder(tf.float32, [None, 784])
# output: 10 classes
y_true = tf.placeholder(tf.float32, [None, 10])
# weight :input: 284, output:  10
W = tf.Variable(tf.zeros([784, 10]))


# 3. 결과를 예측하는 함수를 구성한다.
# y= x * W 
y_pred = tf.matmul(x, W) 

# 4. loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y_pred, labels=y_true))

# 5. optimization
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 6. evaluation 
# argmax : a, axis
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# graph 실행
with tf.Session() as sess:

    # initialization 
    sess.run(tf.global_variables_initializer())

	# 반복 training
    for _ in range(NUM_STEPS):
		#next_batch : 데이터셋으로부터 필요한 만큼의 데이터를 반환하는 함수
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # Evaluation 
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))
