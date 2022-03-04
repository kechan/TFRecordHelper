from __future__ import print_function

from enum import Enum
from re import X
import tensorflow as tf
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# class extension for Path
Path.ls = lambda x: list(x.iterdir())
Path.lf = lambda pth, pat='*': list(pth.glob(pat))    # allow ls with wildchar match
Path.rlf = lambda pth, pat='*': list(pth.rglob(pat))  # allow recursive ls with wildchar match


# recommended helpers from tensorflow doc
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _byteslist_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _floatlist_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class TFRecordHelper:

  class DataType(Enum):
    INT = 1
    FLOAT = 2
    STRING = 3    # can be a "raw string" encoded from an image format like jpeg
    INT_ARRAY = 4
    FLOAT_ARRAY = 5
    STRING_ARRAY = 6
    VAR_STRING_ARRAY = 7

  @staticmethod
  def create(*, feature_desc, feature_constructor=None, dataset, output_filename='test.tfrecords'):
    """
    Create a tfrecord from a tf.data.Dataset that is a dict as item.

    Inputs:

    feature_desc: a dict specifying the "schema" of the tfrecord
    feature_constructor: a dict of key and function to construct that value
    dataset: a tf.data.Dataset that has dict as a single sample 

    """
    
    bad_items = []
    with tf.io.TFRecordWriter(output_filename) as writer:
      for i, x in enumerate(dataset):

        try:
          feature = {}
          for k, v in feature_desc.items():
            if isinstance(v, tuple):
              t, n = v
              assert n == len(feature_constructor[k](x)), f'array is not of length {n}'
              if t == __class__.DataType.INT_ARRAY:
                feature[k] = _int64list_feature(feature_constructor[k](x)) if feature_constructor is not None else _int64list_feature(x[k])
              elif t == __class__.DataType.FLOAT_ARRAY:
                feature[k] = _floatlist_feature(feature_constructor[k](x)) if feature_constructor is not None else _floatlist_feature(x[k])
              else:
                pass

            else:
              if v == __class__.DataType.STRING:
                feature[k] = _bytes_feature(feature_constructor[k](x)) if feature_constructor is not None else _bytes_feature(x[k])
              elif v == __class__.DataType.INT:
                feature[k] = _int64_feature(feature_constructor[k](x)) if feature_constructor is not None else _int64_feature(x[k])
              elif v == __class__.DataType.FLOAT:              
                feature[k] = _float_feature(feature_constructor[k](x)) if feature_constructor is not None else _float_feature(x[k])
              elif v == __class__.DataType.VAR_STRING_ARRAY:
                feature[k] = _byteslist_feature(feature_constructor[k](x)) if feature_constructor is not None else _byteslist_feature(x[k])
              else:
                pass
    
          tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
          writer.write(tf_example.SerializeToString())

          if i % 100 == 0: print(".", end='')
          if i % 1000 == 0: print(i, end='')
          if i % 10000 == 0: print("")

        except Exception as e:
          print(e)  
          bad_items.append(x)

    return bad_items

  @staticmethod
  def parse_fn(feature_desc):
    feature_tf_io_desc = {}

    for k, v in feature_desc.items():
      if isinstance(v, tuple):
        t, n = v
        if t == __class__.DataType.INT_ARRAY:
          feature_tf_io_desc[k] = tf.io.FixedLenFeature([n], tf.int64)
        elif t == __class__.DataType.FLOAT_ARRAY:
          feature_tf_io_desc[k] = tf.io.FixedLenFeature([n], tf.float32)
        else:
          pass
      else:
        if v == __class__.DataType.STRING:
          feature_tf_io_desc[k] = tf.io.FixedLenFeature([], tf.string)
        elif v == __class__.DataType.INT:
          feature_tf_io_desc[k] = tf.io.FixedLenFeature([], tf.int64)
        elif v == __class__.DataType.FLOAT:
          feature_tf_io_desc[k] = tf.io.FixedLenFeature([], tf.float32)
        elif v == __class__.DataType.VAR_STRING_ARRAY:
          feature_tf_io_desc[k] = tf.io.VarLenFeature(tf.string)
        else:
          pass

    def parse_feature_function(example_proto):
      return tf.io.parse_single_example(example_proto, feature_tf_io_desc)

    return parse_feature_function

  @staticmethod
  def shuffle(*, feature_desc, feature_constructor, dataset, seed=None, output_filename='test.tfrecords'):

    len_dataset = len([x for x in dataset])
    bad_items = TFRecordHelper.create(feature_desc, 
                                      feature_constructor, 
                                      dataset.shuffle(len_dataset, seed=seed), 
                                      output_filename=output_filename
                                 )

  @staticmethod
  def element_spec(ds):
    for raw_record in ds.take(1):
      example = tf.train.Example()
      example.ParseFromString(raw_record.numpy())
      return example


class TFRecordHelperWriter(object):
  def __init__(self, filename: str, features: Dict[str, Any]):
    self.filename = filename
    self.features = features
    
  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    #TODO: probably need to handle exception and close any resource
    pass

  def write(self, dataset: tf.data.Dataset, feature_constructors: Optional[Dict[str, Any]] = None):
    print(f'len(dataset): {tf.data.Dataset.cardinality(dataset).numpy()}')
    TFRecordHelper.create(feature_desc = self.features,
                          feature_constructor = feature_constructors,
                          dataset = dataset,
                          output_filename = self.filename
                          )

if __name__ == "__main__":
  #TODO: show example of how to use

  file_ds = tf.data.Dataset.from_tensor_slices(
    {'filename': ['abc.jpg', '123.jpg'],
     'filepath': ["/content/abc.jpg", "/content/123.jpg"]
    }
  )

  data_ds = file_ds.map(lambda x: {'filename': x['filename'], 'image_raw': tf.io.read_file(x['filepath'])})

  features = {
    'filename': TFRecordHelper.DataType.STRING,
    'image_raw': TFRecordHelper.DataType.STRING,   # bytes for the encoded jpeg, png, etc.
  }  

  with TFRecordHelperWriter('my_test.tfrecords', features = features) as f:
    f.write(data_ds)


  # read the image back from data_ds (sanity check)
  parse_fn = TFRecordHelper.parse_fn(features)

  img_ds = tf.data.TFRecordDataset('my_test.tfrecords')\
                .map(parse_fn, num_parallel_calls=AUTO)\
                .map(lambda x: tf.image.decode_jpeg(x['image_raw'], channels=3), num_parallel_calls=AUTO)


  for img in img_ds:
    plt.imshow(img); plt.grid()
    break
