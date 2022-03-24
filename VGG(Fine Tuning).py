import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Dropout, Dense, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt


print(os.listdir("Dataset"))  # The process to print all data  in the dataset
print(len(os.listdir('Dataset/train')), "training data")  # The process to print the training data in the dataset
print(len(os.listdir('Dataset/test')), "test data")  # The process to print the test data in the dataset

# The parameter settings
img_size = 224
input_shape = (img_size, img_size, 3)
epochs = 10
batch_size = 16


# The process to label the cat/dog images: 1 for dog and 0 for cat
def generate_label(directory):
    label = []
    for file in os.listdir(directory):
        if file.split('.')[0] == 'dog':
            label.append(str(1))
        elif file.split('.')[0] == 'cat':
            label.append(str(0))
    return label


#  The process to add the file paths to the columns of our dataframe for loading and training our images
def get_path(directory):
    path = []
    for files in os.listdir(directory):
        path.append(files)
    return path


# The process of formatting data
train_x = get_path('Dataset/train')
train_y = generate_label('Dataset/train')
test_x = get_path('Dataset/test')
df = pd.DataFrame({'filename': train_x,
                   'category': train_y})


# The process to do transfer learning through fine tuning in VGG16 network
pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
# The process to freeze the first 15 layers and only train the last 1 layer in VGG16 network
for layer in pre_trained_model.layers[:15]:
    layer.trainable = False
for layer in pre_trained_model.layers[15:]:
    layer.trainable = True
last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output

# The process to flatten the output layer to 1 dimension
x = GlobalMaxPooling2D()(last_output)
# The process to add fully-connected layer 1 with 512 neurons
x = Dense(512, activation='relu')(x)
# The process of adding dropout function to avoid overfitting
x = Dropout(0.5)(x)
# The process to add fully-connected layer 2 with 1 neuron
x = layers.Dense(1, activation='sigmoid')(x)

# Define training methods:
# Parameters:
# loss function: Use binary cross-entropy loss
# optimizer: Implement Adam optimizer to make training converge faster
# metrics: Set to evaluate model:
model = Model(pre_trained_model.input, x)
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])
model.summary()

# The process to split the data
train_df, valid_df = train_test_split(df, test_size=0.2)
# The process to reset the index of the training set
train_df = train_df.reset_index()
# The process to reset the index of the validation set
valid_df = valid_df.reset_index()
# The process to print the shape of the training dataframe and validation dataframe
total_train = train_df.shape[0]
total_validate = valid_df.shape[0]

# The process of generating mini batches of augmented data in the training set
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "Dataset/train/",
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    validate_filenames=False
)

# The process of generating mini batches of augmented data in the validation set
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    valid_df,
    "Dataset/train/",
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    validate_filenames=False
)

# The process to start training
history = model.fit_generator(train_generator,
                              epochs=epochs,
                              validation_data=validation_generator)
# The process to evaluate the accuracy of the model
loss, accuracy = model.evaluate_generator(validation_generator, validation_generator.samples // batch_size, workers=12)
print("The validation accuracy is %f ; The validation loss is %f " % (accuracy, loss))


# The process to show the training process
def show_train_process(train_accuracy, val_accuracy):
    plt.plot(history.history[train_accuracy])
    plt.plot(history.history[val_accuracy])
    plt.title('The Training history')
    plt.xlabel('The number of Epoch')
    plt.xlim(0, 10)
    plt.ylabel('The accuracy')
    plt.ylim(0, 1)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()


# Plot accuracy rate execution curve
show_train_process('accuracy', 'val_accuracy')
# Plot the loss function execution curve
show_train_process('loss', 'val_loss')
