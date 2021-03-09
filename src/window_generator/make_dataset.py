import tensorflow as tf
import numpy as np

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  batch_size = 33

  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=False,
      batch_size=batch_size,)

  ds = ds.map(self.split_window)

  return ds