
def prepare(tf):
    # fix issue with "cannot find dnn implementation"
    # https://github.com/tensorflow/tensorflow/issues/36508
    physical_device = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(physical_device, enable=True) 
    print("Enabled experimental memory growth for", physical_device)

    print("Tensorflow:", tf.__version__)
