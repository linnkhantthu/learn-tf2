import os
import tensorflow as tf

# Load model path
path = os.path.dirname(os.path.abspath(__file__))
imported = tf.saved_model.load(
    f"{path}/model/1703389724/")
