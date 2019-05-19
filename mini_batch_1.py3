import tensorflow as tf
import os
import keras
import math

DATASET_PATH = 'D:/datasets/cats_and_dogs/train'
DATASET_VALID = 'D:/datasets/cats_and_dogs/valid'
NO_OF_CLASSES = 2
IMG_HEIGHT = 224
IMG_WIDTH = 224
NO_OF_CHANNELS = 3
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
AUTOTUNE = tf.data.experimental.AUTOTUNE

import imp
my_module = imp.load_source('module1',"D:/workspace/my_custom_modules/my_helper.py3")


train_images, train_labels = my_module.loadImagesAndLabels(DATASET_PATH)
valid_images, valid_labels = my_module.loadImagesAndLabels(DATASET_VALID)

def buildDataset(images, labels, batch_size=1):
    #prefetch_op = tf.contrib.data.prefetch_to_device(device="/device:GPU:0")
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    #ds = ds.apply(prefetch_op)
    image_label_ds = ds.map(my_module.load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)
    ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(images)))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE) # The number of elements to prefetch should be equal to (or possibly greater than) the number of batches consumed by a single training step.
    return ds

def performBatchTraining(batch_no, model, train_images, train_labels, valid_images, valid_labels, train_batch_size, valid_batch_size, no_epochs=5):

	print('+++++++++++++++++++++++++++++++++++++++++', batch_no, '+++++++++++++++++++++++++++++++++++++++++')
	
	train_dataset = buildDataset(train_images, train_labels, train_batch_size)
	valid_datdataset = buildDataset(valid_images, valid_labels, valid_batch_size)
	steps_per_epoch = math.ceil(len(train_images) / train_batch_size)
	validation_steps = math.ceil(len(valid_images) / valid_batch_size)
	print("steps_per_epoch :", steps_per_epoch)
	print("validation_steps :", validation_steps)
	with tf.device("/device:GPU:0"):
		model.fit(train_dataset, validation_data=valid_datdataset, validation_steps=validation_steps,steps_per_epoch=steps_per_epoch, epochs=no_epochs, verbose=1)
	return model

def prepareBatch(batch_partitions, model, train_images, train_labels, valid_images, valid_labels, train_batch_size, valid_batch_size, no_epochs=5):

	train_batch_partitions = math.ceil(len(train_images) / batch_partitions)
	valid_batch_partitions = math.ceil(len(valid_images) / batch_partitions)
	print('TRAIN MINIBATCH SIZE: ',train_batch_partitions)
	print('VALID MINIBATCH SIZE: ',valid_batch_partitions)

	train_start_pointer = 0
	train_end_pointer = train_batch_partitions
	valid_start_pointer = 0
	valid_end_pointer = valid_batch_partitions

	for batch_no in range(0, batch_partitions):
		batch_train_images = train_images[train_start_pointer: train_end_pointer]
		batch_train_labels = train_labels[train_start_pointer: train_end_pointer]
		batch_valid_images = valid_images[valid_start_pointer: valid_end_pointer]
		batch_valid_labels = valid_labels[valid_start_pointer: valid_end_pointer]

		model = performBatchTraining(batch_no + 1, model, batch_train_images, batch_train_labels, batch_valid_images, batch_valid_labels, train_batch_size, valid_batch_size)

		del train_images[train_start_pointer: train_end_pointer]
		del train_labels[train_start_pointer: train_end_pointer]
		del valid_images[valid_start_pointer: valid_end_pointer]
		del valid_labels[valid_start_pointer: valid_end_pointer]

	return model



model = my_module.loadMobileNetV1(no_of_classes=NO_OF_CLASSES)
model = prepareBatch(4, model, train_images, train_labels, valid_images, valid_labels, 16, 8)

DATASET_PATH = 'D:/datasets/cats_and_dogs/test'
images, labels = my_module.loadImagesAndLabels(DATASET_PATH)
dataset = buildDataset(images, labels)
tse_loss, tset_accu = model.evaluate(dataset, steps=1)
print(tse_loss)
print(tset_accu)

model.save_weights('D:/workspace/Tensorflow_Tutorial/perfect_mini_batch/saved_weights/first_attempt');
saved_mode_path = tf.keras.experimental.export_saved_model(model, 'D:/workspace/Tensorflow_Tutorial/perfect_mini_batch/saved_models/model_v1')