import os
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

def loadImagesAndLabels(image_path, size_per_class = 0):
    images = []
    labels = []
    label = 0
    print('           ')
    classes = sorted(os.walk(image_path).__next__()[1])
    print('Classes: ', classes)
    image_per_class = 0
    for clas in classes:
        class_per_img_counter = 0
        class_dir = os.path.join(image_path, clas)
        print('Location: ',class_dir)
        filename = os.walk(class_dir).__next__()
        image_per_class = len(filename[2])
        for sample in filename[2]:
            if size_per_class > 0:
                if class_per_img_counter < size_per_class:
                    images.append(os.path.join(class_dir, sample))
                    labels.append(label)
                    class_per_img_counter += 1
                else:
                    break
            else:
                images.append(os.path.join(class_dir, sample))
                labels.append(label)
        label += 1
    new_images = []
    for i in range(0, image_per_class):
        new_images.append(images[i])
        new_images.append(images[i + image_per_class]) # handle here in case of more classes
    images = new_images
    new_labels = []
    for i in range(0, image_per_class):
        new_labels.append(labels[i])
        new_labels.append(labels[i + image_per_class]) # handle here in case of more classes
    labels = new_labels
    print('--------------------------------')
    print('Total images: ',len(images))
    print('Total labels: ', len(labels))
    print('First Image path: ', images[0])
    print('First Image label: ', labels[0])
    print('Last Image path: ', images[-1])
    print('Last Image label: ', labels[-1])
    print('--------------------------------')
    print('           ')
    return (images, labels)

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [224, 224])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image) 

def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

def loadMobileNetV1(no_of_classes, load_previous_weights='', defaultCompile=True):
    preBuiltModel = tf.keras.applications.MobileNet(weights=None, classes=2)
    model = tf.keras.Sequential()
    lenght = len(preBuiltModel.layers)

    for layer in range(lenght):
        model.add(preBuiltModel.layers[layer])
    print(model.summary())

    if load_previous_weights != '':
        model.load_weights(load_previous_weights)

    if defaultCompile:
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

    return model

def convertImgToArray(image_path, image_size):
    im = load_img(image_path)
    im = im.resize((image_size, image_size), Image.ANTIALIAS)
    x = img_to_array(im)  
    x = x / 255.0
    return x

def getImageTensors(images, img_size):
    temp_list = []
    counter = -0;
    for image in images:
        temp_list.append(convertImgToArray(image, img_size))
        counter += 1
        print('Converted ',counter,' out of ',len(images))
    return np.array(temp_list)