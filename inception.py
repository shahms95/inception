from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
import utils
import numpy as np

model = InceptionV3(include_top=True, weights=None)

# print(model.summary())

sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

model.compile(sgd, loss='categorical_crossentropy')

ROOT_DIR = '/var/lib/nova/imagenet/ILSVRC/Data/CLS-LOC/'


train_datagen  = ImageDataGenerator()
test_datagen = ImageDataGenerator()
    
img_rows, img_cols = 299,299
train_generator = train_datagen.flow_from_directory(
        ROOT_DIR + 'train/',
        target_size=(img_rows, img_cols),#The target_size is the size of your input images,every image will be resized to this size
        batch_size=32,
        class_mode='categorical')

print("Train Generator's work is done!")

validation_generator = test_datagen.flow_from_directory(
        ROOT_DIR + 'val/',
        target_size=(img_rows, img_cols),#The target_size is the size of your input images,every image will be resized to this size
        batch_size=32,
        class_mode='categorical')

print("Validation Generator's work is done!")

model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=10, validation_data=validation_generator
        )

# res = model.evaluate(x = np.array(data), y = np.array(label))

# print(res)