from keras.models import load_model
import os
import numpy as np
from dataset import Dataset
import utils
# import keras.layers as KL
# import keras.applications.resnet50 import ResNet50
# from keras.layers import Flatten, Dense, AveragePooling2D
# import keras.models import Model
from keras.models import load_model
test_dataset=Dataset()
test_dataset.load_stan('test')
test_dataset.prepare()

weight_path = 'weight.h5'

# Resnet50_notop = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(448,224,3))
# output = Resnet50_notop.get_layer(index=-1).output
# output = KL.TimeDistributed(AveragePooling2D((7,7), padding='valid', name='avg_pool'))(output)
# output = KL.Reshape((2,1,2048))(output)
# s1 = KL.Lambda(lambda x: x[:, 0, :, :], output_shape=(1, 1, 2048))(output)
# s2 = KL.Lambda(lambda x: x[:, 1, :, :], output_shape=(1, 1, 2048))(output)
# output = KL.Concatenate(axis=-1)([s1,s2])
# # s1 = Flatten(name='flatten_s1')(s1)
# # s2 = Flatten(name='flatten_s2')(s2)
# # s1 = Dense(40, activation='softmax', name='predictions_s1')(s1)
# # s2 = Dense(40, activation='softmax', name='predictions_s2')(s2)
# #output = KL.MaxPooling2D(pool_size=(2,1), padding='valid', name='max_pool')(output)
# output = Flatten(name='flatten')(output)
# logits= Dense(40, name='logits')(output)
# output = KL.Activation('softmax', name='predictions')(logits)
# def action_loss(args):
#     return K.sparse_categorical_crossentropy(args[0], args[1])
# loss = KL.Lambda(lambda x:K.sum(action_loss(x)), name='action_loss')([input_action, output])
# Resnet50_model = Model(Resnet50_notop.input, output)

Action_model = load_model(weight_path)


image_ids = np.copy(test_dataset.image_ids())
sum = len(image_ids)
num =0
for i in image_ids:
    image_concat = utils.load_test(test_dataset, i, augment=True)
    label = test_dataset.load_label(i)
    image_concat = np.reshape(image_concat, [1,448,224,3])
    output= Action_model.predict(image_concat)
    pred = np.argmax(output)
    if pred==label:
        num+=1

print('acc:',num/sum)

