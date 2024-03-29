from __future__ import print_function

import traceback
from enum import Enum
from re import X
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from typing import Any, Callable, List, Dict, Optional


Path.ls = lambda x: list(x.iterdir())
Path.lf = lambda pth, pat='*': list(pth.glob(pat))
Path.rlf = lambda pth, pat='*': list(pth.rglob(pat))


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
    VAR_FLOAT_ARRAY = 8
    VAR_INT_ARRAY = 9

  @staticmethod
  def create(*, feature_desc, feature_constructor=None, dataset, output_filename='test.tfrecords', shard_size: int = None, mask_indice: List[int] = None):
    """
    Create a tfrecord from a tf.data.Dataset that is a dict as item.

    Inputs:

    feature_desc: a dict specifying the "schema" of the tfrecord
    feature_constructor: a dict of key and function to construct that value
    dataset: a tf.data.Dataset that has dict as a single sample 
    output_filename: the output filename
    shard_size: the size of each shard. If None, then no sharding is done.
    mask_indice: a list of indice to mask out. If None, then no masking is done.
    """
    # print(f'{feature_constructor is not None}')
    
    bad_items = []
    # with tf.io.TFRecordWriter(output_filename) as writer:

    writer = tf.io.TFRecordWriter(output_filename)
    for i, x in tqdm(enumerate(dataset)):

      if mask_indice is not None and i in mask_indice: continue

      try:
        feature = {}
        for k, v in feature_desc.items():
          if isinstance(v, tuple):
            t, n = v
            assert n == len(feature_constructor[k](x)) if feature_constructor is not None else len(x[k]), f'array is not of length {n}'
            if t.value == __class__.DataType.INT_ARRAY.value:
              feature[k] = _int64list_feature(feature_constructor[k](x)) if feature_constructor is not None else _int64list_feature(x[k])
            elif t.value == __class__.DataType.FLOAT_ARRAY.value:
              feature[k] = _floatlist_feature(feature_constructor[k](x)) if feature_constructor is not None else _floatlist_feature(x[k])
            elif t.value == __class__.DataType.STRING_ARRAY.value:
              feature[k] = _byteslist_feature(feature_constructor[k](x)) if feature_constructor is not None else _byteslist_feature(x[k])
            else:
              print(f"Serious Warning: failed to write data for {k} and {v}. Please debug.")
              pass

          else:
            if v.value == __class__.DataType.STRING.value:
              feature[k] = _bytes_feature(feature_constructor[k](x)) if feature_constructor is not None else _bytes_feature(x[k])

            elif v.value == __class__.DataType.INT.value:
              feature[k] = _int64_feature(feature_constructor[k](x)) if feature_constructor is not None else _int64_feature(x[k])

            elif v.value == __class__.DataType.FLOAT.value:              
              feature[k] = _float_feature(feature_constructor[k](x)) if feature_constructor is not None else _float_feature(x[k])

            elif v.value == __class__.DataType.VAR_STRING_ARRAY.value:
              feature[k] = _byteslist_feature(feature_constructor[k](x)) if feature_constructor is not None else _byteslist_feature(x[k])

            elif v.value == __class__.DataType.VAR_FLOAT_ARRAY.value:
              feature[k] = _floatlist_feature(feature_constructor[k](x)) if feature_constructor is not None else _floatlist_feature(x[k])

            elif v.value == __class__.DataType.VAR_INT_ARRAY.value:
              feature[k] = _int64list_feature(feature_constructor[k](x)) if feature_constructor is not None else _int64list_feature(x[k])

            else:
              print(f"Serious Warning: failed to write data for {k} and {v}. Please debug.")
              pass
  
        if shard_size is not None and i % shard_size == 0:
          writer.close()
          writer = tf.io.TFRecordWriter(f'{output_filename}.{i // shard_size}')

        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(tf_example.SerializeToString())

        # if i % 100 == 0: print(".", end='')
        # if i % 1000 == 0: print(i, end='')
        # if i % 10000 == 0: print("")

      except Exception as e:
        print(e)  
        print(traceback.format_exc())
        bad_items.append(x)
    
    writer.close()
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
        elif t == __class__.DataType.STRING_ARRAY:
          feature_tf_io_desc[k] = tf.io.FixedLenFeature([n], tf.string)
        else:
          print(f"Serious Warning: failed to read data for {k} and {v}. Please debug.")
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
        elif v == __class__.DataType.VAR_FLOAT_ARRAY:
          feature_tf_io_desc[k] = tf.io.VarLenFeature(tf.float32)
        elif v == __class__.DataType.VAR_INT_ARRAY:
          feature_tf_io_desc[k] = tf.io.VarLenFeature(tf.int64)
        else:
          print(f"Serious Warning: failed to read data for {k} and {v}. Please debug.")
          pass

    def parse_feature_function(example_proto):
      return tf.io.parse_single_example(example_proto, feature_tf_io_desc)

    return parse_feature_function

  @staticmethod
  def shuffle(*, feature_desc, feature_constructor, dataset, seed=None, output_filename='test.tfrecords'):

    len_dataset = len([x for x in dataset])
    bad_items = TFRecordHelper.create(feature_desc = feature_desc, 
                                      feature_constructor = feature_constructor, 
                                      dataset = dataset.shuffle(len_dataset, seed=seed), 
                                      output_filename=output_filename
                                 )

  @staticmethod
  def element_spec(ds, return_keys_only=False):
    for raw_record in ds.take(1):
      example = tf.train.Example()
      example.ParseFromString(raw_record.numpy())
      
      if return_keys_only:
        feature_keys = list(example.features.feature.keys())
        return feature_keys

      return example 

  @staticmethod
  def count(ds):
    for k, _ in enumerate(ds):
      pass
    return k + 1


class TFRecordHelperWriter(object):
  def __init__(self, 
    filename: str, 
    features: Dict[str, Any], 
    shard_size: Optional[int] = None, 
    mask_indice: Optional[List[int]] = None):

    '''
    filename: output filename
    features: a dictionary of [string: feature_type] pairs, where feature_type can be one of any TFRecordHelper.DataType.*
    shard_size: if specified, will shard the output into multiple files of size shard_size
    mask_indice: if specified, will skip writing record included in the mask indices. Useful for writing a subset of the dataset
    '''

    self.filename = filename
    self.features = features
    self.shard_size = shard_size
    self.mask_indice = mask_indice
    
  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    #TODO: probably need to handle exception and close any resources
    pass

  def write(self, dataset: tf.data.Dataset, feature_constructors: Optional[Dict[str, Any]] = None):
    print(f'len(dataset): {tf.data.Dataset.cardinality(dataset).numpy()}')
    TFRecordHelper.create(feature_desc = self.features,
                          feature_constructor = feature_constructors,
                          dataset = dataset,
                          output_filename = self.filename,
                          shard_size = self.shard_size,
                          mask_indice = self.mask_indice
                          )

if __name__ == "__main__":
  #TODO: show example of how to use
  import matplotlib.pyplot as plt

  file_ds = tf.data.Dataset.from_tensor_slices(
    {'filename': ['9_Ropewalk_Lane_Dartmouth_NS.jpg', '31_Saddlebrook_Way_NE_Calgary_AB.jpg'],
     'filepath': ["/content/9_Ropewalk_Lane_Dartmouth_NS.jpg", "/content/31_Saddlebrook_Way_NE_Calgary_AB.jpg"]
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
                .map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)\
                .map(lambda x: tf.image.decode_jpeg(x['image_raw'], channels=3), num_parallel_calls=tf.data.AUTOTUNE)


  for img in img_ds:
    plt.imshow(img); plt.grid()
    break
