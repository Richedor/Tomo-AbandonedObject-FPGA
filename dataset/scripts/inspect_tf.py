import tensorflow as tf, sys, os
print('TF_VERSION:', getattr(tf,'__version__','n/a'))
print('HAS_tf_lite:', hasattr(tf,'lite'))
if hasattr(tf,'lite'):
    print('HAS_Interpreter:', hasattr(tf.lite,'Interpreter'))
print('TensorFlow dir sample:', sorted(list(dir(tf)))[:40])
