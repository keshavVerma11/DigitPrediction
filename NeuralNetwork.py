import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu

def neural_network(X,y):
    tf.random.set_seed(1234) # for consistent results
    model = Sequential(
        [               
            Dense(25, activation = 'relu', name = 'layer1'),
            Dense(15, activation = 'relu', name = 'layer2'),
            Dense(10, activation = 'linear', name = 'layer3')
        ], name = "digit_prediction" 
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    
    model.fit(
        X,y,
        epochs=40
    )
    return model

def prediction(model, index, X):
    prediction = model.predict(X[index].reshape(1,400))
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)
    return yhat