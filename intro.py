import tensorflow as tf

# Version
print(tf.__version__)

# Creating tensors(numpy arrays)
rank_1 = tf.Variable(["Hello", "World"], tf.string)
rank_2 = tf.Variable([["MALE", "YES"], ["FEMALE", "NO"]], tf.string)

print(rank_1)
print(rank_2)
