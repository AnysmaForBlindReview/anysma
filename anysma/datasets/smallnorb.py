"""smallNORB dataset for classification.

URL:
    https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/

LICENCE / TERMS / COPYRIGHT:
    This database is provided for research purposes. It cannot be sold. Publications that include results obtained with
    this database should reference the following paper:

    Y. LeCun, F.J. Huang, L. Bottou, Learning Methods for Generic Object Recognition with Invariance to Pose and
    Lighting. IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR) 2004

Description:
    This database is intended for experiments in 3D object reocgnition from shape. It contains images of 50 toys
    belonging to 5 generic categories: four-legged animals, human figures, airplanes, trucks, and cars. The objects
    were imaged by two cameras under 6 lighting conditions, 9 elevations (30 to 70 degrees every 5 degrees), and 18
    azimuths (0 to 340 every 20 degrees).

    The training set is composed of 5 instances of each category (instances 4, 6, 7, 8 and 9), and the test set of the
    remaining 5 instances (instances 0, 1, 2, 3, and 5).

Workaround copied from Keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils.data_utils import get_file
import numpy as np


def load_data(path='smallnorb.npz'):
    """Loads the smallnorb dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train, info_train), (x_test, y_test, info_test)`.
    """
    path = get_file(path,
                    origin='http://tiny.cc/anysma_datasets_smallnorb',
                    file_hash='8c0e51ad773fa13a9352c097c45ff9c6')
    f = np.load(path)
    x_train, y_train, info_train = f['x_train'], f['y_train'], f['info_train']
    x_test, y_test, info_test = f['x_test'], f['y_test'], f['info_test']
    f.close()
    return (x_train, y_train, info_train), (x_test, y_test, info_test)
