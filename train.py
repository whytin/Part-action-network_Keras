from keras.applications.resnet50 import ResNet50
#from resnet50 import ResNet50
import os
import keras.layers as KL
import keras.backend as K
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from dataset import Dataset
import utils
import logging

learning_rate = 0.0001
image_width = 224
image_height = 224
train_samples = 300
validation_samples = 50
epochs = 100
batch_size=16

train_data_dir = '/usr/local/datasets/stanford40/keras_train'
val_data_dir = '/usr/local/datasets/stanford40/keras_val'
actions=['applauding', 'blowing_bubbles', 'brushing_teeth',
                               'cleaning_the_floor', 'climbing', 'cooking',
                               'cutting_trees', 'cutting_vegetables', 'drinking',
                               'feeding_a_horse', 'fishing', 'fixing_a_bike',
                               'fixing_a_car', 'gardening', 'holding_an_umbrella',
                               'jumping', 'looking_through_a_microscope', 'looking_through_a_telescope',
                               'playing_guitar', 'playing_violin', 'pouring_liquid', 'pushing_a_cart',
                               'reading', 'phoning', 'riding_a_bike', 'riding_a_horse',
                               'rowing_a_boat', 'running', 'shooting_an_arrow', 'smoking',
                               'taking_photos', 'texting_message', 'throwing_frisby', 'using_a_computer',
                               'walking_the_dog', 'washing_dishes', 'watching_TV', 'waving_hands',
                               'writing_on_a_board', 'writing_on_a_book']

print('Loading Resnet50 Weights ...')
#input_action = KL.Input(shape=[None], name='input_labels')
Resnet50_notop = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(448,224,3))
Resnet50_notop.trainable=False
print('Adding Average Pooling Layer and Softmax Output Layer ...')

output = Resnet50_notop.get_layer(index=-1).output
output = KL.TimeDistributed(AveragePooling2D((7,7), padding='valid', name='avg_pool'))(output)
output = KL.Reshape((2,1,2048))(output)
s1 = KL.Lambda(lambda x: x[:, 0, :, :], output_shape=(1, 1, 2048))(output)
s2 = KL.Lambda(lambda x: x[:, 1, :, :], output_shape=(1, 1, 2048))(output)
output = KL.Concatenate(axis=-1)([s1,s2])
# s1 = Flatten(name='flatten_s1')(s1)
# s2 = Flatten(name='flatten_s2')(s2)
# s1 = Dense(40, activation='softmax', name='predictions_s1')(s1)
# s2 = Dense(40, activation='softmax', name='predictions_s2')(s2)
#output = KL.MaxPooling2D(pool_size=(2,1), padding='valid', name='max_pool')(output)
output = Flatten(name='flatten')(output)
logits= Dense(40, name='logits')(output)
output = KL.Activation('softmax', name='predictions')(logits)
# def action_loss(args):
#     return K.sparse_categorical_crossentropy(args[0], args[1])
# loss = KL.Lambda(lambda x:K.sum(action_loss(x)), name='action_loss')([input_action, output])
#input=[Resnet50_notop.input, input_action]
#Resnet50_model = Model(input, output)
Resnet50_model = Model(Resnet50_notop.input, output)


optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov= True)
#optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#Resnet50_model.add_loss(loss)
Resnet50_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

best_model_file='./weight.h5'
best_model=ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only=True)

train_dataset=Dataset()
train_dataset.load_stan('train')
train_dataset.prepare()


val_dataset=Dataset()
val_dataset.load_stan('test')
val_dataset.prepare()



def data_generator(dataset, shuffle=True, augment=True, batch_size=32):
    b = 0
    image_index = -1
    image_ids = np.copy(dataset.image_ids())
    #print(image_ids)
    error_count = 0
    while True:
        try:
            #print(b)
            image_index = (image_index+1) % len(image_ids)
            if shuffle and image_index==0:
                np.random.shuffle(image_ids)
            image_id = image_ids[image_index]
            image_concat = utils.load_train(dataset, image_id, augment=augment)
            label = dataset.load_label(image_id)
            if b==0:
                batch_images = np.zeros((batch_size,448,224,3), dtype=np.float32)
                batch_actions = np.zeros((batch_size,1), dtype=np.int32)

            batch_images[b] = image_concat
            batch_actions[b] = label

            b +=1
            if b>=batch_size:
                #inputs=[batch_images, batch_actions]
                inputs=batch_images
                outputs=batch_actions
                yield inputs, outputs
                b=0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise



train_generator=data_generator(train_dataset, shuffle=True, augment=True, batch_size=16)
validation_generator=data_generator(val_dataset, shuffle=True, augment=True, batch_size=16)

#print(train_dataset.image_ids())
#print(val_dataset.image_ids())

# train_datagen=ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.1,
#     zoom_range=0.1,
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True
# )

# val_datagen=ImageDataGenerator(rescale=1./255)
#
# train_generator=train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(448, 224),
#     batch_size=batch_size,
#     shuffle=True,
#     classes=actions,
#     class_mode='categorical'
# )
#
# validation_generator=val_datagen.flow_from_directory(
#     val_data_dir,
#     target_size=(448, 224),
#     batch_size=batch_size,
#     shuffle=True,
#     classes=actions,
#     class_mode='categorical'
# )

Resnet50_model.fit_generator(
    train_generator,
    steps_per_epoch= train_samples,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps = validation_samples,
    callbacks=[best_model]
)
