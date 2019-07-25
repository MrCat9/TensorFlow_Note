# -*- coding: utf-8 -*-


import dataset_test_set
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed

seed(10)

set_random_seed(20)

batch_size = 400

# Prepare input data
classes = ['dogs', 'cats']
num_classes = len(classes)

img_size = 64
num_channels = 3
test_path = 'testing_data'

data = dataset_test_set.read_test_sets(test_path, img_size, classes)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in testing-set:\t\t{}".format(len(data.test.labels)))

#

x_batch, y_test_batch, _, y_test_true_cls = data.test.next_batch(batch_size)

# Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('./dogs-cats-model/dog-cat.ckpt-7950.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, './dogs-cats-model/dog-cat.ckpt-7950')

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

# Let's feed the images to the input placeholders
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")

# Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {x: x_batch, y_true: y_test_batch}
y_pred = sess.run(y_pred, feed_dict=feed_dict_testing)
# print(y_pred)

res_label = ['dogs', 'cats']
correct_prediction_num = 0
for i in range(y_pred.shape[0]):
    p = res_label[y_pred[i, :].argmax()]
    t = y_test_true_cls[i]
    if p == t:
        correct_prediction_num += 1
    print(i, p, t)
acc = correct_prediction_num / y_pred.shape[0]
print(acc)
