import tensorflow as tf
print('TF_VERSION', getattr(tf,'__version__','n/a'))
print('HAS_LITE', hasattr(tf,'lite'))
print('HAS_INTERPRETER', hasattr(tf.lite, 'Interpreter') if hasattr(tf,'lite') else False)
