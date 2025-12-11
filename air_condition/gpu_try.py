import tensorflow as tf
print("TF version:", tf.__version__)
print("Physical devices:", tf.config.list_physical_devices())
print("GPUs:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA?:", tf.test.is_built_with_cuda())
print("GPU device name (if any):", tf.test.gpu_device_name())
print("TF version:", tf.__version__)
