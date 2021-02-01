import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Layer

class IdentityBlock(tf.keras.Model):
  def __init__(self,filters,kernel_size):
    super(IdentityBlock, self).__init__(name='')
    self.conv1 = tf.keras.layers.Conv2D(filters,kernel_size,padding='same')
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv2D(filters,kernel_size,padding='same')
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.act = tf.keras.layers.Activation('relu')
    self.add = tf.keras.layers.Add()
  
  def call(self,input_tensor):
    x = self.conv1(input_tensor)
    x = self.bn1(x)
    x = self.act(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.act(x)

    x = self.add([x,input_tensor])
    x = self.act(x)
    return x
    
class ResNet(tf.keras.Model):
  def __init__(self,num_classes):
    super(ResNet,self).__init__()
    self.conv = tf.keras.layers.Conv2D(64,7,padding='same')
    self.bn = tf.keras.layers.BatchNormalization()
    self.act = tf.keras.layers.Activation('relu')
    self.max_pool = tf.keras.layers.MaxPool2D((3,3))

    self.idia = IdentityBlock(64,3)
    self.idib = IdentityBlock(64,3)

    self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
    self.classifier = tf.keras.layers.Dense(num_classes, activation='relu')
  
  def call(self,inputs):
    x = self.conv(inputs)
    x = self.bn(x)
    x = self.act(x)
    x = self.max_pool(x)

    x = self.idia(x)
    x = self.idib(x)

    x = self.global_pool(x)

    return self.classifier(x)

# utility function to normalize the images and return (image, label) pairs.
def preprocess(features):
  return tf.cast(features['image'], tf.float32) / 255., features['label']
# create a ResNet instance with 10 output units for MNIST
resnet = ResNet(10)
resnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
dataset = tfds.load('mnist', split= tfds.Split.TRAIN, data_dir='./data')
dataset = dataset.map(preprocess).batch(32)

resnet.fit(dataset,epochs=1)
