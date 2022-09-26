# TFRecordHelper
Helper methods to make manipulating and using TFRecords a bit more friendlier

## Installation

### Install as a python package

Install using pip with the git repo:

```bash
pip install git+https://github.com/kechan/TFRecordHelper
```

## Usage

### Using the Python interface

If you installed the package, you can use it as follows:

```python
from tfrecord_helper import TFRecordHelper, TFRecordHelperWriter

import tensorflow as tf
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
                .map(parse_fn, num_parallel_calls=AUTO)\
                .map(lambda x: tf.image.decode_jpeg(x['image_raw'], channels=3), num_parallel_calls=AUTO)


  for img in img_ds:
    plt.imshow(img); plt.grid()
    break
```
