from keras.models import load_model
import tensorflow as tf
global graph

model = load_model('model.h5')
model._make_predict_function()
graph = tf.get_default_graph()
