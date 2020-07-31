"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_basic_data.py
DESC:     new basic class of class Tensor and Scalar, imported by api/debug
          module
CREATED:  2019-08-14 15:45:18
MODIFIED: 2019-08-14 15:45:18
"""
class BasicData():
    """
    content :
        self.data_type
    """
    def __init__(self, data_type):
        """
        Tensor/Scalar register initialization, recording the data_type of
        BasicData: "Scalar" or "Tensor

        Parameters
        ----------
        data_type: data type of Basic Data

        Returns
        ----------
        None
        """
        self.data_type = data_type

    def is_scalar(self):
        """
        check whether Basic is Scalar

        Returns
        ----------
        bool, whether BasicData is Scalar
        """
        return self.data_type == "Scalar"

    def is_tensor(self):
        """
        check whether Basic is Tensor

        Returns
        ----------
        bool, whether BasicData is Tensor
        """
        return self.data_type == "Tensor"
