import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import tensorflow as tf

print(os.listdir("Dataset"))  # The process to print all data  in the dataset
print(len(os.listdir('Dataset/train')), "training data")  # The process to print the training data in the dataset
print(len(os.listdir('Dataset/test')), "test data")  # The process to print the test data in the dataset

# The parameter settings
img_size = 224
input_shape = (img_size, img_size, 3)
batch_size = 16
epochs = 20


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
# The process of visualizing data distribution
df = pd.DataFrame({'filename': train_x,
                   'category': train_y})
print(df.head())
sns.countplot(x='category', data=df).set_title("Data Distribution")
plt.show()

# The process to split the data
train_df, valid_df = train_test_split(df, test_size=0.2)
# The process to reset the index of the training set
train_df = train_df.reset_index()
# The process to reset the index of the validation set
valid_df = valid_df.reset_index()
# The process to print the shape of the training dataframe and validation dataframe
print(train_df.shape)
print(valid_df.shape)
total_train = train_df.shape[0]
total_validate = valid_df.shape[0]
print(total_train)
print(total_validate)

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

# Construct the Sequential model
model = Sequential()
# Add the first convolution layer
model.add(Conv2D(16, (3, 3), activation="relu", input_shape=(img_size, img_size, 3)))
# Add the second convolution layer
model.add(Conv2D(16, (3, 3), activation="relu", ))
# Add the first pooling layer
model.add(MaxPooling2D((3, 3)))
# Add the third convolution layer
model.add(Conv2D(32, (3, 3), activation="relu"))
# Add the fourth convolution layer
model.add(Conv2D(32, (3, 3), activation="relu"))
# Add the second pooling layer
model.add(MaxPooling2D(2, 2))
# Add the fifth convolution layer
model.add(Conv2D(64, (3, 3), activation="relu"))
# Add the sixth convolution layer
model.add(Conv2D(64, (3, 3), activation="relu"))
# Add the third pooling layer
model.add(MaxPooling2D(2, 2))
# Add dropout function to avoid overfitting
model.add(Dropout(0.3))
# Add the seventh convolution layer
model.add(Conv2D(32, (3, 3), activation="relu"))
# Add the fourth pooling layer
model.add(MaxPooling2D((2, 2)))
# Change multi-dimensional array to one dimension
model.add(Flatten())
# Add fully-connected layer 1 with 512 neurons
model.add(Dense(512, activation="relu"))
# Add dropout function to avoid overfitting
model.add(Dropout(0.5))
# Add fully-connected layer 2 with 1 neuron
model.add(Dense(1, activation="sigmoid"))
# Show model summary
print(model.summary())

# Define training methods:
# Parameters:
# loss function: Use binary cross-entropy loss
# optimizer: Implement Adam optimizer to make training converge faster
# metrics: Set to evaluate model:
model.compile(loss=tf.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=["accuracy"])
# The process to start training
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=validation_generator,
                    shuffle=True)
# The process to evaluate the accuracy of the model
loss, accuracy = model.evaluate_generator(validation_generator, validation_generator.samples // batch_size, workers=12)
print("The validation accuracy is %f ; The validation loss is %f " % (accuracy, loss))


# The process to show the training process
def show_train_process(train_accuracy, val_accuracy):
    plt.plot(history.history[train_accuracy])
    plt.plot(history.history[val_accuracy])
    plt.title('The Training history')
    plt.xlabel('The number of Epoch')
    plt.xlim(0, 20)
    plt.ylabel('The accuracy')
    plt.ylim(0, 1)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()


# Plot accuracy rate execution curve
show_train_process('accuracy', 'val_accuracy')
# Plot the loss function execution curve
show_train_process('loss', 'val_loss')
