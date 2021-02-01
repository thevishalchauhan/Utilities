#Example1
model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=(28,28)),
     tf.keras.layers.Dense(128),
     tf.keras.layers.Lambda(lambda x:tf.abs(x)),
     tf.keras.layers.Dense(10,activation='softmax')])

#Example2
def my_relu(x):
  return K.maximum(-0.1,x)

model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=(28,28)),
     tf.keras.layers.Dense(128),
     tf.keras.layers.Lambda(my_relu),
     tf.keras.layers.Dense(10, activation='softmax')])

#Example3
#Custom Layer
#y = 2x-1
from tensorflow.keras.layers import Layer

class SimpleDense(Layer):

  def __init__(self,units=32):
    super(SimpleDense,self).__init__()
    self.units = units

  def build(self,input_shape):

    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(name='kernel',
                         initial_value = w_init(shape = (input_shape[-1], self.units),
                                                dtype='float32'),
                          trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(name='bias',
                         initial_value = b_init(shape=(self.units,), dtype='float32'),
                         trainable=True)
  def call(self,inputs):
    return tf.matmul(inputs,self.w) + self.b
 
my_dense = SimpleDense(units=1)

x = tf.ones((1,1))
y = my_dense(x)

print(my_dense.variables)

 #Example4
class SimpleDense2(Layer):
  def __init__(self,units=32,activation=None):
    super(SimpleDense2,self).__init__()
    self.units = units
    self.activation =tf.keras.activations.get(activation)

  def build(self, input_shape):
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(name='kernel',
                         initial_value = w_init(shape=(input_shape[-1],self.units),
                                                dtype='float32'),
                         trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(name='bias',
                         initial_value=b_init(shape=(self.units,),dtype='float32'),
                         trainable=True)
    super().build(input_shape)
  
  def call(self,inputs):
    return self.activation(tf.matmul(inputs,self.w)+self.b)
model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=(28,28)),
     SimpleDense2(128,activation='relu'),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(10,activation='softmax')])
     
#Example5
class SimpleQuadratic(Layer):
  def __init__(self, units=32, activation=None):
    '''Initializes the class and sets up the internal variables'''
    super(SimpleQuadratic,self).__init__()
    self.units = units
    self.activation = tf.keras.activations.get(activation)
  def build(self, input_shape):
    '''Create the state of the layer (weights)'''
    # a and b should be initialized with random normal, c (or the bias)␣,!with zeros.
    # remember to set these as trainable.
    a_int = tf.random_normal_initializer()
    a_init_val = a_int(shape = (input_shape[-1], self.units),dtype='float32')
    self.a = tf.Variable(name="a", initial_value=a_init_val, trainable=True)
    b_init = tf.random_normal_initializer()
    b_init_val = b_init(shape= (input_shape[-1],self.units), dtype ='float32')
    self.b = tf.Variable(name="b", initial_value=b_init_val, trainable=True)
    c_init = tf.zeros_initializer()
    c_init_val = c_init(shape = (self.units), dtype='float32')
    self.c = tf.Variable(name='c',initial_value=c_init_val, trainable=True)
  def call(self, inputs):
    '''Defines the computation from inputs to outputs'''
    x_squared = tf.math.square(inputs)
    x_squared_times_a = tf.matmul(x_squared, self.a)
    x_times_b = tf.matmul(inputs,self.b)
    x2a_plus_xb_plus_c = x_squared_times_a + x_times_b + self.c
    return self.activation(x2a_plus_xb_plus_c)
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
SimpleQuadratic(128, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10, activation='softmax')
])
