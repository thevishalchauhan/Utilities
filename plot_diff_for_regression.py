def plot_diff(y_true,y_pred, title=''):
  plt.scatter(y_true, y_pred)
  plt.title(title)
  plt.xlabel('True Value')
  plt.ylabel('Predictions')
  plt.axis('equal')
  plt.axis('square')
  plt.plot([-100,100],[-100,100])
  return plt
