# -*- coding: utf-8 -*-

from typing import Union

import torch
import tensorflow as tf

Tensor = Union[torch.Tensor, tf.Tensor, tf.Variable]
