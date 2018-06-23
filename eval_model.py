from __future__ import print_function


from tensorflow.python.layers import core as layers_core
from utils.iterator_utils import get_iterator
from utils.embedding_utils import create_embedding_layer_for_encoder_and_decoder as create_embedding
from decoder.decoder import *
from train_model import BaseModel
import tensorflow as tf
import os

BASE = "/home/lizijian/dataset/en-vi/"


class EvalModel(BaseModel):

    def __init__(self, iterator, src_vocab_size=17191, tgt_vocab_size=7709,
                 src_embedding_size=512, tgt_embedding_size=512, mode=tf.contrib.learn.ModeKeys.EVAL,
                 unit_type="lstm", num_unit=512, num_layers=2, forget_bias=1.0, dropout=0.2):

        if mode == tf.contrib.learn.ModeKeys.EVAL:
            self.mode = mode
        else:
            raise ValueError("TrainingModel with %s mode" % mode)

        self.iterator = iterator
        self.src_ids, self.tgt_input_ids, self.tgt_output_ids, self.src_seq_len, self.tgt_seq_len \
            = self.iterator.get_next()

        self.batch_size = tf.size(self.src_seq_len)

        initial_weight = 0.1
        variables_initializer = tf.random_uniform_initializer(-initial_weight, initial_weight)
        tf.get_variable_scope().set_initializer(variables_initializer)

        self.embedding_encoder, self.embedding_decoder = create_embedding(src_embedding_size=src_embedding_size,
                                                                          tgt_embedding_size=tgt_embedding_size,
                                                                          src_vocab_size=src_vocab_size,
                                                                          tgt_vocab_size=tgt_vocab_size)

        self.output_layer = layers_core.Dense(tgt_vocab_size, use_bias=False, name="procjection_layer")

        self.decoder_output, self.decoder_state = self.create_training_model(mode=mode)

        with tf.variable_scope("decoder"):
            self.logits = self.output_layer(self.decoder_output)

        self.eval_loss = self._compute_loss(logits=self.logits)
        self.predict_count = tf.reduce_sum(self.tgt_seq_len)

        self.global_step = tf.Variable(0, trainable=False)

        self.saver = tf.train.Saver(tf.global_variables())

        parameters = tf.trainable_variables()
        for param in parameters:
            print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                    param.op.device))

    def eval(self, eval_sess):
        return eval_sess.run([self.eval_loss, self.predict_count, self.batch_size])
        pass

    def get_logits(self, eval_sess):
        return eval_sess.run(self.logits)
        pass


if __name__ == "__main__":
    src_dataset_name = "tst2012.en"
    tgt_dataset_name = "tst2012.vi"
    src_vocab_name = "vocab.en"
    tgt_vocab_name = "vocab.vi"

    src_dataset_path = os.path.join(BASE, src_dataset_name)
    tgt_dataset_path = os.path.join(BASE, tgt_dataset_name)
    src_vocab_path = os.path.join(BASE, src_vocab_name)
    tgt_vocab_path = os.path.join(BASE, tgt_vocab_name)

    batch_size = 128
    num_buckets = 5

    iterator = get_iterator(src_dataset_path=src_dataset_path, tgt_dataset_path=tgt_dataset_path,
                            src_vocab_path=src_vocab_path, tgt_vocab_path=tgt_vocab_path,
                            batch_size=batch_size, num_buckets=num_buckets, is_shuffle=False,
                            src_max_len=None, tgt_max_len=None)

    model = EvalModel(iterator=iterator)

    session = tf.Session()
    session.run(tf.tables_initializer())
    session.run(tf.global_variables_initializer())
    session.run(iterator.initializer)

    for i in range(1000):
        print(i)
        eval_loss, predict_count, batch_size = model.eval(eval_sess=session)
    # print(eval_loss)
    # print(predict_count)
    # print(batch_size)
    # print(model.decoder_output)

    pass