from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
import utils
import numpy as np

model = InceptionV3(include_top=True, weights=None)

# print(model.summary())

sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

model.compile(sgd, loss='categorical_crossentropy')

d = utils.get_data('../../../var/lib/nova/imagenet/ILSVRC/Data/CLS-LOC/train')

data = d['data']
label = d['label']
model.fit(x = data, y = label, epochs=10)

# res = model.evaluate(x = np.array(data), y = np.array(label))

# print(res)