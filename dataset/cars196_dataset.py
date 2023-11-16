# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 19:02:49 2016

@author: sakurai
"""

from fuel.datasets import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from lib import nn_Ops
import copy

class Cars196Dataset(H5PYDataset):

    _filename = 'cars196/cars196.hdf5'

    def __init__(self, which_sets, **kwargs):
        try:
#           path = find_in_data_path(self._filename)
            path = "/home/ZSL/workspace/HDML/datasets/data/cars196/cars196.hdf5"
        except IOError as e:
            msg = str(e) + (""".
         You need to download the dataset and convert it to hdf5 before.""")
            raise IOError(msg)
        super(Cars196Dataset, self).__init__(
            file_or_path=path, which_sets=which_sets, **kwargs)


def load_as_ndarray(which_sets=['train', 'test']):
    datasets = []
    for split in which_sets:
        data = Cars196Dataset([split], load_in_memory=True).data_sources
        datasets.append(data)
    return datasets


if __name__ == '__main__':
    dataset = Cars196Dataset(['train'])

    st = DataStream(
        dataset, iteration_scheme=SequentialScheme(dataset.num_examples, 1))
    it = st.get_epoch_iterator()
    for batch in copy.copy(it):
        # get images and labels from batch
        x_batch_data, Label_raw = nn_Ops.batch_data(batch)
