import numpy as np
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

hvd.init()

train_data = np.load(f'data_train_{hvd.rank()}.npz')
x_train, y_train = train_data['x_train'], train_data['y_train']

model = tf.keras.models.Sequential([
  layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
  layers.BatchNormalization(),
  layers.MaxPool2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.GlobalAveragePooling2D(),
  layers.Dropout(0.2),
  layers.Dense(64, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(10, activation='softmax')
])

opt = hvd.DistributedOptimizer(tf.optimizers.Adam(0.01), backward_passes_per_step=1, average_aggregated_gradients=True)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

datagen = ImageDataGenerator(horizontal_flip=True)
model.fit(datagen.flow(x_train, y_train, batch_size=8), callbacks=callbacks, epochs=3, verbose=(hvd.rank() == 0))

if hvd.rank() == 0:
    test_data = np.load('data_test.npz')
    x_test, y_test = test_data['x_test'], test_data['y_test']
    preds = model.predict(x_test)
    acc_score = accuracy_score(y_test[:, 0], np.argmax(preds, axis=1))
    print(f'Model accuracy is {acc_score}')
