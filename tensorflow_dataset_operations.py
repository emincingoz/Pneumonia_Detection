import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class TensorflowDatasetClass:
  def __init__(self, image_size, batch_size = 32):
    self.image_size = image_size
    self.batch_size = batch_size

  def get_data_tensorflow_dataset(self, path):
    image_set = keras.preprocessing.image_dataset_from_directory(directory = path, 
                                                                color_mode = 'grayscale', 
                                                                seed = 135, 
                                                                image_size = self.image_size, 
                                                                batch_size = self.batch_size, 
                                                                shuffle = True)
    return image_set

  def get_shapes(self, image_dataset):
    image_shape_count = 0
    label_shape_count = 0
    image_shape = 0
    labels_shape = 0
    for image_batch, labels_batch in image_dataset:
      image_shape_count += image_batch.shape[0]
      label_shape_count += labels_batch.shape[0]
      image_shape = image_batch.shape
      labels_shape = labels_batch.shape

    y = list(image_shape)
    y[0] = image_shape_count
    image_shape = tuple(y)

    y = list(labels_shape)
    y[0] = label_shape_count
    labels_shape = tuple(y)

    
    print('image set shape: ', image_shape)
    print('labels set shape: ', labels_shape)
        
  def get_class_names(self, image_dataset):
    return image_dataset.class_names

  def get_images_and_labels_as_nparray(self, image_dataset):
    image_list = []
    labels_list = []
    for image_batch, labels_batch in image_dataset:
      image_list.append(image_batch)
      labels_list.append(labels_batch)

    image_list = np.array(image_list)
    #image_list = image_list[:, :, :, :, 0] # 5043, 1, 180, 180
    labels_list = np.array(labels_list)

    # You can use numpy.squeeze() to remove all dimensions of size 1 from the NumPy array ndarray. 
    # squeeze() is also provided as a method of ndarray.
    image_list = np.squeeze(image_list)
    labels_list = np.squeeze(labels_list)

    if self.batch_size == 1:
      image_list = np.squeeze(image_list)
      labels_list = np.squeeze(labels_list)

      return np.array(image_list), np.array(labels_list)
    
    # returns [images, height, width]
    return np.array(np.squeeze(image_list[0])), np.array(np.squeeze(labels_list[0]))

  def take_samples(self, image_dataset, image_count):
    class_names = self.get_class_names(image_dataset)
    images, labels = self.get_images_and_labels_as_nparray(image_dataset)

    plt.figure(figsize=(10, 10))
    for i in range(image_count):
      plt.subplot(3, 3, i + 1)
      if self.batch_size == 1:
        plt.imshow(images[i].astype("uint8"), cmap = 'gray')
      else:
        plt.imshow(images[i].astype('uint8'), cmap = 'gray')
      plt.title(class_names[labels[i]])
      plt.axis("off")