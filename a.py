import tensorflow as tf
print("TF version:", tf.__version__)
print("Built with CUDA:", tf.sysconfig.get_build_info().get("cuda_version", "N/A") if hasattr(tf.sysconfig, "get_build_info") else "N/A")
print("CUDA visible devices:", tf.config.list_physical_devices('GPU'))



