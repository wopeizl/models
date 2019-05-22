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
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import multiprocessing
import sys
from PIL import Image
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC
from paddle.fluid.dygraph.base import to_variable
import nets
import reader
from config import SentaConfig
from utils import ArgumentGroup, print_arguments
from utils import init_checkpoint

DATA_PATH="./senta_data/"
CKPT_PATH="./save_models"
MODEL_PATH="./save_models/step_1800/"
VOCAB_PATH=DATA_PATH + "word_dict.txt"
SENTA_CONFIG_PATH="./senta_config.json"
skip_steps =10

parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("senta_config_path", str, SENTA_CONFIG_PATH, "Path to the json file for senta model config.")
model_g.add_arg("init_checkpoint", str, CKPT_PATH, "Init checkpoint to resume training from.")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 10, "Number of epoches for training.")
train_g.add_arg("save_steps", int, 5000, "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 1000, "The steps interval to evaluate model performance.")
train_g.add_arg("lr", float, 0.002, "The Learning rate value for training.")

log_g = ArgumentGroup(parser, "logging", "logging related")
log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
log_g.add_arg("verbose", bool, False, "Whether to output verbose log")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_dir", str, DATA_PATH, "Path to training data.")
data_g.add_arg("vocab_path", str, VOCAB_PATH, "Vocabulary path.")
data_g.add_arg("batch_size", int, 256, "Total examples' number in batch for training.")
data_g.add_arg("random_seed", int, 0, "Random seed.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")
run_type_g.add_arg("task_name", str, "cnn_net",
    "The name of task to perform sentiment classification.")
run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
run_type_g.add_arg("do_val", bool, True, "Whether to perform evaluation.")
run_type_g.add_arg("do_infer", bool, True, "Whether to perform inference.")

args = parser.parse_args()

args.batch_size = 20
padding_size = 150

senta_config = SentaConfig(args.senta_config_path)

if args.use_cuda:
    place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    dev_count = fluid.core.get_cuda_device_count()
else:
    place = fluid.CPUPlace()
    dev_count = 2


def reader_decorator(reader):
    def _reader_imple():
        for item in reader():
            doc = np.pad(item[0][0:padding_size], (0, padding_size - len(item[0][0:padding_size])), 'constant').astype('int64').reshape(padding_size, 1)
            label = np.array(item[1]).reshape(1)
            yield doc, label

    return _reader_imple


def train_cnn():
    with fluid.dygraph.guard():
        processor = reader.SentaProcessor(data_dir=args.data_dir,
                                          vocab_path=args.vocab_path,
                                          random_seed=args.random_seed)
        num_labels = len(processor.get_labels())

        num_train_examples = processor.get_num_examples(phase="train")

        max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        train_data_generator = processor.data_generator(
            batch_size=args.batch_size,
            phase='train',
            epoch=args.epoch,
            shuffle=True)

        eval_data_generator = processor.data_generator(
            batch_size=args.batch_size,
            phase='dev',
            epoch=args.epoch,
            shuffle=False)

        py_reader = fluid.io.PyReader()
        py_reader.decorate_sample_list_generator(
                paddle.batch(
                    reader_decorator(paddle.dataset.mnist.train()),
                    batch_size=args.batch_size,
                    drop_last=True),
                places=fluid.CPUPlace())

        cnn_net = nets.CNN("cnn_net", 33256, args.batch_size, padding_size)
        # train_reader.decorate_sample_list_generator(
        #     train_data_generator, places=fluid.CPUPlace())

        sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=args.lr)
        steps = 0
        total_cost, total_acc, total_num_seqs = [], [], []

        for eop in range(args.epoch):
            time_begin = time.time()
            # for batch_id, data in enumerate(py_reader()):
            for batch_id, data in enumerate(train_data_generator()):
                steps += 1
                np_doc = np.array(
                        [np.pad(x[0][0:padding_size], (0,padding_size - len(x[0][0:padding_size])), 'constant') for x in data]).astype('int64').reshape(1, -1)
                label = to_variable(np.array(
                        [x[1] for x in data]).astype('int64').reshape(args.batch_size, 1))

                # mask = np_doc > 0
                # assert (np.array_equal(np_doc, np_doc * mask))
                # print(mask)

                # print(len(mask))
                # mask = np.repeat(mask, 128, axis=0).reshape(-1, 128)
                # for i in range(7):
                #     mask = np.concatenate((mask, mask))
                # mask = np.transpose(mask)
                # print(mask.shape)
                # print(mask)
                # # os._exit(1)
                # # print(len(mask))
                # mask = to_variable(np.array(mask, dtype='float32'))

                doc = to_variable(np_doc.reshape(-1, 1))
                mask = None
                # doc = data[0]
                # label = data[1]
                # print(doc, label)
                cnn_net.train()
                avg_cost, prediction, acc = cnn_net(doc, label, mask)
                avg_cost.backward()
                sgd_optimizer.minimize(avg_cost)

                # num_seqs = fluid.layers.create_tensor(dtype='int64')
                # accuracy = fluid.layers.accuracy(input=prediction, label=label, total=num_seqs)

                dy_out = avg_cost.numpy()
                cnn_net.clear_gradients()

                total_cost.append(avg_cost.numpy())
                total_acc.append(acc.numpy())
                total_num_seqs.append(1)
                # print(total_cost, total_acc, total_num_seqs)
                # print(len(total_cost), len(total_acc), len(total_num_seqs))
                # print("epoch id: %d, batch step: %d, loss: %f" % (eop, batch_id, dy_out))

                if steps % args.skip_steps == 0:

                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("step: %d, ave loss: %f, "
                        "ave acc: %f, speed: %f steps/s" %
                        (steps, np.sum(total_cost) / np.sum(total_num_seqs),
                        np.sum(total_acc) / np.sum(total_num_seqs),
                        args.skip_steps / used_time))
                    total_cost, total_acc, total_num_seqs = [], [], []
                    time_begin = time.time()

                # if steps % args.save_steps == 0:
                #     fluid.dygraph.save_persistables(cnn_net.state_dict(),
                #                                     "save_dir_" + str(steps))

                if steps % args.validation_steps == 0:
                    total_eval_cost, total_eval_acc, total_eval_num_seqs = [], [], []
                    cnn_net.eval()
                    eval_steps = 0
                    for eval_batch_id, eval_data in enumerate(eval_data_generator()):
                        eval_np_doc = np.array(
                            [np.pad(x[0][0:padding_size], (0, padding_size - len(x[0][0:padding_size])), 'constant') for
                             x in eval_data]).astype('int64').reshape(1, -1)
                        eval_label = to_variable(np.array(
                            [x[1] for x in eval_data]).astype('int64').reshape(args.batch_size, 1))
                        eval_doc = to_variable(eval_np_doc.reshape(-1, 1))
                        eval_mask = None

                        eval_avg_cost, eval_prediction, eval_acc = cnn_net(eval_doc, eval_label, eval_mask)
                        total_eval_cost.append(eval_avg_cost.numpy())
                        total_eval_acc.append(eval_acc.numpy())
                        total_eval_num_seqs.append(1)
                        eval_steps += 1

                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("Final validation result:")
                    print(" step: %d, ave loss: %f, "
                        "ave acc: %f, speed: %f steps/s" %
                        (steps, np.sum(total_eval_cost) / np.sum(total_eval_num_seqs),
                        np.sum(total_eval_acc) / np.sum(total_eval_num_seqs),
                        eval_steps / used_time))

                    time_begin = time.time()

def eval_cnn():
    with fluid.dygraph.guard():
        cnn_net = nets.CNN("cnn_net", 33256, args.batch_size, padding_size)
        fluid.dygraph.save_persistables(cnn_net.state_dict(),
                                    "save_dir")


if __name__ == '__main__':
    train_cnn()
