import sys

def main():
    print('Python:', sys.version)
    try:
        import tensorflow as tf
        v = getattr(tf, '__version__', 'NO_VERSION_ATTR')
        print('TensorFlow import OK, version =', v)
    except Exception as e:
        print('TensorFlow import FAILED:', repr(e))
    try:
        import ultralytics
        print('Ultralytics import OK, version =', getattr(ultralytics, '__version__', 'unknown'))
    except Exception as e:
        print('Ultralytics import FAILED:', repr(e))

if __name__ == '__main__':
    main()
