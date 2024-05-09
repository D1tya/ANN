import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

def baseline_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    return model

baseline_model = baseline_model()

baseline_model.compile(optimizer='adam',
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])

baseline_history = baseline_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

test_loss, test_accuracy = baseline_model.evaluate(x_test, y_test)
print("Baseline Model Test Accuracy:", test_accuracy)


def tuned_model(dropout_rate=0.5, learning_rate=0.001):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])

    model.compile(optimizer=optimizers.Adam(lr=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

tuned_model = tuned_model(dropout_rate=0.3, learning_rate=0.0001)
tuned_history = tuned_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

test_loss, test_accuracy = tuned_model.evaluate(x_test, y_test)
print("Tuned Model Test Accuracy:", test_accuracy)

plt.plot(baseline_history.history['accuracy'], label='Baseline Training Accuracy')
plt.plot(baseline_history.history['val_accuracy'], label='Baseline Validation Accuracy')
plt.plot(tuned_history.history['accuracy'], label='Tuned Training Accuracy')
plt.plot(tuned_history.history['val_accuracy'], label='Tuned Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.plot(baseline_history.history['loss'], label='Baseline Training Loss')
plt.plot(baseline_history.history['val_loss'], label='Baseline Validation Loss')
plt.plot(tuned_history.history['loss'], label='Tuned Training Loss')
plt.plot(tuned_history.history['val_loss'], label='Tuned Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


