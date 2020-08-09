try:
    import matplotlib.pyplot
except ImportError:
    print("conv_visualiser has a requirement for matplotlib")

try:
    import numpy
except ImportError:
    print("conv_visualiser has a requirement for numpy")

try:
    import tensorflow as tf
    if tf.__version__ <= str(2.0):
        print("Incompatible version cnn_visualiser has a requirement of "
              "tensorflow version 2.0 or greater")
        raise ImportError
except ImportError:
    print("conv_visualiser has a requirement for tensorflow")

from . import cnn_visualiser
