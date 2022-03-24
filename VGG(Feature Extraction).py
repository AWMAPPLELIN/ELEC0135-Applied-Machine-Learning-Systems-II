import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from hypopt import GridSearch
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg16 import VGG16

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
# The process of visualizing data distribution
df = pd.DataFrame({'filename': train_x,
                   'category': train_y})
sns.countplot(x='category', data=df).set_title("Data Distribution")
os.chdir('Dataset/train')
img = load_img(df['filename'].iloc[0])
plt.figure(figsize=(8, 8))
plt.imshow(img)

# The process to split the data
train_df, valid_df = train_test_split(df, test_size=0.2)

# The process of generating mini batches of augmented data in the training set
train_datagen = ImageDataGenerator(
    rotation_range=10,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df[['filename']],
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    validate_filenames=False
)

# The process of generating mini batches of augmented data in the validation set
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    valid_df,
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    validate_filenames=False
)

# The process to do transfer learning through feature extraction in VGG16 network
vgg = VGG16(weights='imagenet',
            include_top=False,  # The process to remove the fully connected layers from the pretrained VGG16 model
            input_shape=(224, 224, 3))
vgg.summary()
for layers in vgg.layers:
    layers.trainable = False
print(vgg.output)

# The process to create feature list
feature_list = []
for path in train_df['filename'].to_numpy():
    x = load_img(path, target_size=(img_size, img_size))
    img_array = img_to_array(x)
    img_array = np.expand_dims(img_array, axis=0)
    features = vgg.predict(img_array)
    feature_list.append(features)
feat_lst = np.reshape(feature_list, (-1, 7 * 7 * 512))
y = train_df['category'].to_numpy()

# The process to split the data
X_train, X_valid, y_train, y_valid = train_test_split(feat_lst, y, test_size=0.2, random_state=2020)
# The process to search the best parameter with the function parameter_grid and GridSearch
param_grid = [{'C': [0.1, 1, 10], 'solver': ['newton-cg', 'lbfgs']}]
optimizer = GridSearch(model=LogisticRegression(class_weight='balanced', multi_class="auto",
                                                max_iter=200, random_state=1), param_grid=param_grid)
# The process to start training
optimizer.fit(X_train, y_train)
# The process to evaluate the accuracy of the model
print("The Accuracy on validation set using Logistic Regression is : ", optimizer.score(X_valid, y_valid))
