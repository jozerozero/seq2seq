from __future__ import print_function

import os

from cell.cell import *
from encoder.encoder import *
from tensorflow.python.layers import core as layers_core
from decoder.decoder import *
from utils.embedding_utils import create_embedding_layer_for_encoder_and_decoder as create_embedding
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils.iterator_utils import get_iterator

BASE = "/home/lizijian/dataset/en-vi/"


class BaseModel(object):

    def __init__(self):
        self.iterator = None
        self.src_ids = None
        self.tgt_input_ids = None
        self.tgt_output_ids = None
        self.src_seq_len = None
        self.tgt_seq_len = None
        self.embedding_encoder = None
        self.embedding_decoder = None
        self.batch_size = None
        self.encoder_output = None
        self.encoder_state = None

        raise ValueError("Please implement the BaseModel")
        pass

    def _compute_loss(self, logits):
        target_outputs = self.tgt_output_ids
        target_outputs = tf.transpose(target_outputs)
        """max_time represent the max time step of the target sentence"""
        max_time =target_outputs.shape[0].value

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_outputs, logits=logits)

        target_sequence_mask_vectors = tf.sequence_mask(self.tgt_seq_len, max_time, dtype=logits.dtype)

        target_sequence_mask_vectors = tf.transpose(target_sequence_mask_vectors)

        loss = tf.reduce_sum(cross_entropy * target_sequence_mask_vectors) / tf.to_float(self.batch_size)

        return loss

    def create_training_model(self, mode):
        """the initialization of the iterator should be moved to other, for debuging I put it here first"""

        encoder_output, encoder_state = self.create_encoder_model(mode=mode)

        self.encoder_output = encoder_output
        self.encoder_state = encoder_state

        decoder_output, decoder_state = self.create_decoder_model(mode=mode,
                                                                  encoder_output=encoder_output,
                                                                  encoder_state=encoder_state)

        return decoder_output, decoder_state

    def create_encoder_model(self, mode, unit_type="lstm", num_unit=512, num_layers=2,
                             forget_bias=1.0, dropout=0.2, encode_type="bi"):

        """use the unidirectional rnn as the encoder"""
        if encode_type == "uni":
            encoder_cell = build_rnn_cell(mode=mode, unit_type=unit_type, num_unit=num_unit,
                                        forget_bias=forget_bias, dropout=dropout, num_layers=num_layers)

            encoder_output, encoder_state = build_uni_encoder(num_layer=num_layers, source_tokens=self.src_ids,
                                                              source_length=self.src_seq_len, cell=encoder_cell,
                                                              embedding_encoder=self.embedding_encoder)

        elif encode_type == "bi":
            """use the bidirectional rnn as the encoder"""
            encoder_cell = build_rnn_cell(mode=mode, unit_type=unit_type, num_unit=num_unit,
                                          forget_bias=forget_bias, dropout=dropout, num_layers=num_layers/2)

            encoder_output, encoder_state = build_bi_encoder(num_layer=num_layers, source_tokens=self.src_ids,
                                                             source_length=self.src_seq_len, cell=encoder_cell,
                                                             embedding_encoder=self.embedding_encoder)
        else:
            raise ValueError("Unknow % encode tpye" % encode_type)

        return encoder_output, encoder_state

    def create_decoder_model(self, mode, encoder_output, encoder_state, unit_type="lstm", num_unit=512, num_layers=2,
                             forget_bias=1.0, dropout=0.2, use_attention=True):

        decoder_cell = build_rnn_cell(mode=mode, unit_type=unit_type, num_unit=num_unit,
                                      forget_bias=forget_bias, dropout=dropout, num_layers=num_layers)

        if not use_attention:
            print("not use attention")
            decoder_output, decoder_state = build_original_decoder(cell=decoder_cell, encoder_output=encoder_output,
                                                                   encoder_state=encoder_state,
                                                                   target_sequence_length=self.tgt_seq_len,
                                                                   source_sequence_length=self.src_seq_len,
                                                                   target_tokens=self.tgt_input_ids,
                                                                   embedding_decoder=self.embedding_decoder)
        else:
            print("use attention")
            decoder_output, decoder_state = build_attention_decoder(cell=decoder_cell, encoder_output=encoder_output,
                                                                    encoder_state=encoder_state,
                                                                    source_sequence_length=self.src_seq_len,
                                                                    target_sequence_length=self.tgt_seq_len,
                                                                    target_tokens=self.tgt_input_ids,
                                                                    embedding_decoder=self.embedding_decoder,
                                                                    batch_size=self.batch_size, num_units=num_unit)
        return decoder_output, decoder_state


class TrainingModel(BaseModel):

    def __init__(self, iterator, src_vocab_size=17191, tgt_vocab_size=7709,
                 src_embedding_size=512, tgt_embedding_size=512, mode=tf.contrib.learn.ModeKeys.TRAIN,
                 unit_type="lstm", num_unit=512, num_layers=2, forget_bias=1.0, dropout=0.2,
                 start_decay_step=8000, learning_rate=1.0, decay_steps=1000, decay_factor=0.5):

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.mode = mode
        else:
            raise ValueError("TrainingModel with %s mode" % mode)

        self.iterator = iterator
        self.src_ids, self.tgt_input_ids, self.tgt_output_ids, self.src_seq_len, self.tgt_seq_len\
            = self.iterator.get_next()

        self.batch_size = tf.size(self.src_seq_len)
        # self.batch_size = tf.constant(128)
        # print(batch_size)
        initial_weight = 0.1
        variables_initializer = tf.random_uniform_initializer(-initial_weight, initial_weight)
        tf.get_variable_scope().set_initializer(variables_initializer)

        self.embedding_encoder, self.embedding_decoder = create_embedding(src_embedding_size=src_embedding_size,
                                                                          tgt_embedding_size=tgt_embedding_size,
                                                                          src_vocab_size=src_vocab_size,
                                                                          tgt_vocab_size=tgt_vocab_size)
        """
        projection layer of decoder, in another word this is the last layer of the model, whose size equals to the 
        size of target vocabulary size.
        """
        # with tf.variable_scope("build_network"):
        #     with tf.variable_scope("decoder/output_projection"):
        self.output_layer = layers_core.Dense(tgt_vocab_size, use_bias=False, name="procjection_layer")

        """the return of the create_training_model might be changed"""
        self.decoder_output, self.decoder_state = self.create_training_model(mode=mode)

        with tf.variable_scope("decoder"):
            self.logits = self.output_layer(self.decoder_output)

        self.train_loss = self._compute_loss(logits=self.logits)

        self.word_count = tf.reduce_sum(self.src_seq_len) + tf.reduce_sum(self.tgt_seq_len)
        self.predict_count = tf.reduce_sum(self.tgt_seq_len)

        self.global_step = tf.Variable(0, trainable=False)

        parameters = tf.trainable_variables()

        self.learning_rate = tf.cond(self.global_step < start_decay_step, lambda: tf.constant(learning_rate),
                                     lambda: tf.train.exponential_decay(learning_rate=learning_rate,
                                                                        global_step=self.global_step - start_decay_step,
                                                                        decay_steps=decay_steps,
                                                                        decay_rate=decay_factor,
                                                                        staircase=True), name="learning")
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        """I am afraid I can't tell you the meaning of colocate_gradients_with_ops, cause I don't know"""
        gradients = tf.gradients(self.train_loss, parameters, colocate_gradients_with_ops=True)
        """gradient clip code"""
        max_gradient_norm = 5
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)

        self.update = optimizer.apply_gradients(zip(clipped_gradients, parameters), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

        for param in parameters:
            print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                              param.op.device))

    def train(self, train_sess):
        return train_sess.run([self.update, self.train_loss, self.predict_count,
                               self.global_step, self.word_count, self.batch_size])
        # return train_sess.run([self.update, self.global_step, self.src_ids])

if __name__ == "__main__":

    src_dataset_name = "train.en"
    tgt_dataset_name = "train.vi"
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
                            batch_size=batch_size, num_buckets=num_buckets, is_shuffle=False)

    model = TrainingModel(iterator=iterator)

    session = tf.Session()
    session.run(tf.tables_initializer())
    session.run(tf.global_variables_initializer())
    session.run(model.iterator.initializer)

    # print(test_result[0])
    # print("====")
    # print(test_result[2])
    print(model.train(session))
    for i in range(3200):
        if i % 200 == 0:
            print(i)
        try:
            # _, global_steps, src_id = model.train(session)
            # session.run([model.decoder_state, model.decoder_output])
            decoder_output, batch_size = session.run([model.decoder_output, model.batch_size])
            decoder_output = decoder_output.transpose([1, 0, 2])
            print(decoder_output.shape)
            # print(decoder_output.shape)
            break
            if batch_size != 128:
                print("batch size : %d, global step : %d" % (batch_size, i+1))
        except :
            session.run(model.iterator.initializer)
            # print(e.message)
            print("run out %d" % (i+1))
    exit()
    # print(training_loss)
    # print(predict_count)
    # print(global_step)
    # print(word_count)
    # print(batch_size)
    # print(logits.shape)
    # '''the shape of the encoder is [batch_size, encoder_output_size]'''
    # print(encoder_state[0].c.shape)