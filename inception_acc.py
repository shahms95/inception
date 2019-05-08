import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
import numpy as np
import CallBack
import argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", nargs='*', type=int, default=0,
                    help="The IDs of GPUs to be used; default = 0")
parser.add_argument("-b", "--bsize", type=int, default=32,
                    help="The Batch size to be used; default = 32")
args = parser.parse_args()

print("List of GPUs specified : ", args.gpu)
# config = tf.ConfigProto( device_count = {'GPU': args.gpu } ) 


config = tf.ConfigProto() 

cvd= str(args.gpu[0])

for i in range(1,len(args.gpu)):
    cvd = cvd + "," + args.gpu[i]

# os.environ["CUDA_VISIBLE_DEVICES"]=cvd
os.environ["CUDA_VISIBLE_DEVICES"]='0'
print("CUDA visible devices : ", os.environ["CUDA_VISIBLE_DEVICES"])

config.gpu_options.allow_growth=True

sess = tf.Session(config=config) 

K.set_session(sess)

model = InceptionV3(include_top=True, weights=None)

# print(model.summary())

sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

ROOT_DIR = '../imagenet-project/ILSVRC/Data/CLS-LOC/'


train_datagen  = ImageDataGenerator()
test_datagen = ImageDataGenerator()
    
img_rows, img_cols = 299,299 # 299x299 for inception, 224x224 for VGG and Resnet

train_generator = train_datagen.flow_from_directory(
        ROOT_DIR + 'train/',
        target_size=(img_rows, img_cols),#The target_size is the size of your input images,every image will be resized to this size
        batch_size=args.bsize,
        class_mode='categorical')

print("Train Generator's work is done!")

validation_generator = test_datagen.flow_from_directory(
        ROOT_DIR + 'val/',
        target_size=(img_rows, img_cols),#The target_size is the size of your input images,every image will be resized to this size
        batch_size=args.bsize,
        class_mode='categorical')

print("Validation Generator's work is done!")


train_gens = tf.split(train_generator, len(args.gpu))
validation_gens = tf.split(validation_generator, len(args.gpu))

# for i in len(args.gpu):

# for i in range(len(args.gpu)):
    # print("Entering GPU : {}".format(args.gpu[i]))
# with tf.device('/gpu:%d'%args.gpu[i] ):
# with tf.device('/gpu:0' ):
history = model.fit_generator(
        train_gens[i],
        steps_per_epoch=2000,
        epochs=12, validation_data=validation_gens[i],
        validation_steps=50,
        verbose = 1
        )

# print(history.history)
# print(history.epoch)
with tf.device('/cpu:0'):
    with open('inception_acc.txt', 'w+') as f:
        for i1, i2, i3 in zip(history.epoch, history.history['acc'], history.history['loss']):
            f.write(str(i1))
            f.write(', ')
            f.write(str(i2))
            f.write(', ')
            f.write(str(i3))
            f.write('\n')