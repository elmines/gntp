# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class SRUFusedRNN(tf.compat.v1.nn.rnn_cell.RNNCell):
    """Simple Recurrent Unit, very fast.  https://openreview.net/pdf?id=rJBiunlAW"""

    def __init__(self, num_units, f_bias=1.0, r_bias=0.0, with_residual=True):
        self._num_units = num_units
        cell = _SRUUpdateCell(num_units, with_residual)
        self._cell = cell
        self._constant_bias = [0.0] * self._num_units + [f_bias] * self._num_units
        if with_residual:
            self._constant_bias += [r_bias] * self._num_units

        self._constant_bias = np.array(self._constant_bias, np.float32)
        self._with_residual = with_residual

    def __call__(self, inputs, initial_state=None, dtype=tf.float32, sequence_length=None, scope=None):
        num_gates = 3 if self._with_residual else 2
        transformed = tf.keras.layers.Dense(
            num_gates * self._num_units,
            bias_initializer=tf.constant_initializer(self._constant_bias),
        )(inputs)

        gates = tf.split(transformed, num_gates, axis=2)
        forget_gate = tf.sigmoid(gates[1])
        transformed_inputs = (1.0 - forget_gate) * gates[0]
        if self._with_residual:
            residual_gate = tf.sigmoid(gates[2])
            inputs *= (1.0 - residual_gate)
            new_inputs = tf.concat([inputs, transformed_inputs, forget_gate, residual_gate], axis=2)
        else:
            new_inputs = tf.concat([transformed_inputs, forget_gate], axis=2)

        return tf.compat.v1.nn.dynamic_rnn(
            self._cell,
            new_inputs,
            sequence_length=sequence_length,
            initial_state=initial_state,
            dtype=dtype,
            time_major=True,
            scope=scope,
        )


class _SRUUpdateCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """Simple Recurrent Unit, very fast.  https://openreview.net/pdf?id=rJBiunlAW"""

    def __init__(self, num_units, with_residual, activation=None, reuse=None):
        super(_SRUUpdateCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._with_residual = with_residual
        self._activation = activation or tf.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Simple recurrent unit (SRU)."""
        if self._with_residual:
            base_inputs, transformed_inputs, forget_gate, residual_gate = tf.split(inputs, 4, axis=1)
            new_state = forget_gate * state + transformed_inputs
            new_h = residual_gate * self._activation(new_state) + base_inputs
        else:
            transformed_inputs, forget_gate = tf.split(inputs, 2, axis=1)
            new_state = new_h = forget_gate * state + transformed_inputs
        return new_h, new_state
