#Option 1
from tensorflow.keras.losses import Loss
class MyHuberLoss(Loss):

  def __init__(self,threshold=1):
    super().__init__()
    self.threshold = threshold
  
  def call(self, y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= self.threshold
    small_error_loss = tf.square(error) / 2
    big_error_loss = self.threshold * (tf.abs(error) - (0.5 * self.threshold))
    return tf.where(is_small_error, small_error_loss, big_error_loss)

#model.compile(optimizer='sgd', loss=MyHuberLoss(threshold=1.02))

#Option2
def my_huber_loss_with_threshold(threshold):
  def my_huber_loss(y_true,y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error)/2
    big_error_loss = threshold * (tf.abs(error) - (0.5*threshold))

    return tf.where(is_small_error, small_error_loss, big_error_loss)
  return my_huber_loss
#model.compile(optimizer='sgd', loss=my_huber_loss_with_threshold(threshold=1.2))

#Option3
def huber_loss(y_pred,y_true):
  threshold=1
  error = y_true - y_pred
  is_small_error = tf.abs(error) <= threshold
  small_error_loss = tf.square(error)/2
  big_error_loss = threshold* (tf.abs(error) - (0.5 * threshold))
  return tf.where(is_small_error, small_error_loss,big_error_loss)
model.compile(optimizer='sgd', loss=huber_loss)
