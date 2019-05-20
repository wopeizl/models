# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.nn import Embedding

# def cnn_net(data,
#             label,
#             dict_dim,
#             emb_dim=128,
#             hid_dim=128,
#             hid_dim2=96,
#             class_dim=2,
#             win_size=3,
#             is_infer=False):
#     """
#     Conv net
#     """
#     # embedding layer
#     emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
#
#     # convolution layer
#     conv_3 = fluid.nets.sequence_conv_pool(
#         input=emb,
#         num_filters=hid_dim,
#         filter_size=win_size,
#         act="tanh",
#         pool_type="max")
#
#     # full connect layer
#     fc_1 = fluid.layers.fc(input=[conv_3], size=hid_dim2)
#     # softmax layer
#     prediction = fluid.layers.fc(input=[fc_1], size=class_dim, act="softmax")
#     if is_infer:
#         return prediction
#     cost = fluid.layers.cross_entropy(input=prediction, label=label)
#     avg_cost = fluid.layers.mean(x=cost)
#     acc = fluid.layers.accuracy(input=prediction, label=label)
#
#     return avg_cost, prediction

class SimpleConvPool(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=1,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleConvPool, self).__init__(name_scope)

        self._conv2d = Conv2D(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
            self.full_name(),
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class CNN(fluid.dygraph.Layer):
    def __init__(self, name_scope, dict_dim, batch_size, seq_len):
        super(CNN, self).__init__(name_scope)
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.hid_dim2 = 96
        self.class_dim = 2
        self.win_size = 3
        self.batch_size = batch_size
        self.seq_len = seq_len
        init_scale = 0.1
        self.embedding = Embedding(self.full_name(),
                                   size=[self.dict_dim, self.emb_dim],
                                   dtype='float32',
                                   is_sparse=False,
                                   param_attr=fluid.ParamAttr(
                                       name='embedding_para',
                                       initializer=fluid.initializer.UniformInitializer(
                                           low=-init_scale, high=init_scale)))

        self._simple_conv_pool_1 = SimpleConvPool(
            self.full_name(), 1, self.hid_dim, self.win_size, 2, 2, act="tanh")

        # pool_1_shape = self.win_size * 4 * 4
        # scale1 = (2.0 / (pool_1_shape**2 * self.hid_dim))**0.5

        self._fc1 = FC(self.full_name(),
                        size=self.hid_dim2,
                        act="softmax")

        # scale2 = (2.0 / (pool_1_shape**2 * self.class_dim))**0.5
        self._fc_prediction = FC(self.full_name(),
                       size=self.class_dim,
                        act="softmax")

    def forward(self, inputs, label=None):
        # embedding layer
        # print(inputs)
        # inputs = fluid.layers.pad(inputs, [0, 0], 0)
        emb = self.embedding(inputs)
        # print(emb)
        # print([-1, self.seq_len, 1, self.hid_dim])
        # emb = fluid.layers.reshape(
        #     emb, shape=[-1, self.seq_len, 1, self.hid_dim])
        # print(emb[0])
        emb = fluid.layers.reshape(
            emb, shape=[-1, 1, self.seq_len, self.hid_dim])
        print(emb)
        conv_3 = self._simple_conv_pool_1(emb)
        fc_1 = self._fc1(conv_3)
        prediction = self._fc_prediction(fc_1)

        cost = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc = fluid.layers.accuracy(input=prediction, label=label)

        return avg_cost, prediction, acc

