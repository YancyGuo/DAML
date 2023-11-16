# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 19:02:49 2016

@author: sakurai
"""

from fuel.datasets import H5PYDataset
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream

from flags.FLAGS import *
from lib import nn_Ops
import copy


class ProstateDataset(H5PYDataset):

    # _filename = 'prostate/prostate.hdf5'

    def __init__(self, which_sets, **kwargs):
        try:
#           path = find_in_data_path(self._filename)
            path = FLAGS.path
            # path = '/home/ZSL/workspace/HDML/datasets/data/prostate/prostate.hdf5'
        except IOError as e:
            msg = str(e) + (""".
         You need to download the dataset and convert it to hdf5 before.""")
            raise IOError(msg)
        super(ProstateDataset, self).__init__(
            file_or_path=path, which_sets=which_sets, **kwargs)


def load_as_ndarray(which_sets=['train', 'test']):
    datasets = []
    for split in which_sets:
        data = ProstateDataset([split], load_in_memory=True).data_sources
        datasets.append(data)
    return datasets


import scipy.misc

if __name__ == '__main__':
    dataset = ProstateDataset(['test'])
    dataset_train = ProstateDataset(['train'])
    st = DataStream(
        dataset, iteration_scheme=SequentialScheme(dataset.num_examples, 1))
    it = st.get_epoch_iterator()
    cp = copy.copy(it)
    id=0
    for batch in cp:
        # get images and labels from batch
        x_batch_data, Label_raw = nn_Ops.batch_data(batch)
        import scipy.misc
        import matplotlib.pyplot as plt
        img = x_batch_data[0]
        scipy.misc.imsave(str(id)+'.jpg', img)
        plt.imshow(img)
        plt.show()
        id +=1
