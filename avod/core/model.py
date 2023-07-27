"""Abstract detection model.  抽象的检测模型。

This file defines a generic base class for detection models.  Programs that are
designed to work with arbitrary detection models should only depend on this
class.  We intend for the functions in this class to follow tensor-in/tensor-out
design, thus all functions have tensors or lists/dictionaries holding tensors as
inputs and outputs.

Abstractly, detection models predict output tensors given input images
which can be passed to a loss function at training time or passed to a
postprocessing function at eval time. The postprocessing happens outside the
model.

这个文件为检测模型定义了一个通用基类。被设计用于任意检测模型的程序应该只依赖于这个类。
我们打算让这个类中的函数遵循张量输入/张量输出的设计，因此所有函数都有张量或包含张量的
列表/字典作为输入和输出。

抽象地说，检测模型预测给定输入图像的输出张量，这些张量可以在训练时传递给损失函数，
也可以在计算时传递给后处理函数。后处理发生在模型之外。
"""
from abc import ABCMeta
from abc import abstractmethod


class DetectionModel(object):
    """Abstract base class for detection models."""
    __metaclass__ = ABCMeta

    def __init__(self, model_config):
        """Constructor.

        Args:
            model_config: configuration for the model
        """
        self._config = model_config

    @property
    def model_config(self):
        return self._config

    @abstractmethod
    def create_feed_dict(self):
        """ To be overridden
        Creates a feed_dict that can be passed into a tensorflow session

        Returns: a dictionary with tensors as keys and numpy arrays as values
        """
        return dict()

    @abstractmethod
    def loss(self, prediction_dict):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Calling this function requires that groundtruth tensors have been
        provided via the provide_groundtruth function.

        Args:
          prediction_dict: a dictionary holding predicted tensors

        Returns:
          a dictionary mapping strings (loss names) to scalar tensors
            representing loss values.
        """
        pass
