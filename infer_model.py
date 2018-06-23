from __future__ import print_function

from cell.cell import *
from encoder.encoder import *
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import lookup_ops
from decoder.decoder import *
from utils.embedding_utils import create_embedding_layer_for_encoder_and_decoder as create_embedding
from utils.iterator_utils import get_inference_iterator
from utils.vocab_utils import *

BASE = "/home/lizijian/dataset/en-vi/"


class InferModel(object):

    def __init__(self, iterator, tgt_vocab_path, src_vocab_size=17191, tgt_vocab_size=7709,
                 src_embedding_size=512, tgt_embedding_size=512, mode=tf.contrib.learn.ModeKeys.INFER,
                 unit_type="lstm", num_unit=512, num_layers=2, forget_bias=1.0, dropout=0.2,
                 start_decay_step=8000, learning_rate=1.0, decay_steps=1000, decay_factor=0.5, beam_width=10):

        if mode == tf.contrib.learn.ModeKeys.INFER:
            self.mode = mode
        else:
            raise ValueError("TrainingModel with %s mode" % mode)

        self.beam_width = beam_width
        self.iterator = iterator
        self.src_ids, self.src_seq_len = iterator.get_next()

        self.tgt_vocab_table = create_src_vocab_table(src_vocab_path=tgt_vocab_path)

        self.batch_size = tf.size(self.src_seq_len)

        initial_weight = 0.1
        variables_initializer = tf.random_uniform_initializer(-initial_weight, initial_weight)
        tf.get_variable_scope().set_initializer(variables_initializer)

        self.embedding_encoder, self.embedding_decoder = create_embedding(src_embedding_size=src_embedding_size,
                                                                          tgt_embedding_size=tgt_embedding_size,
                                                                          src_vocab_size=src_vocab_size,
                                                                          tgt_vocab_size=tgt_vocab_size)

        with tf.variable_scope("build_network"):
            with tf.variable_scope("output_projection"):
                self.output_layer = layers_core.Dense(tgt_vocab_size, use_bias=False, name="procjection_layer")

        self.sample_id, self.final_context_state = self.create_inference_model(mode=self.mode)

        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(tgt_vocab_path, default_value=UNK)

        self.sample_word = reverse_tgt_vocab_table.lookup(tf.to_int64(self.sample_id))

        self.saver = tf.train.Saver(tf.trainable_variables())

        self.global_step = tf.Variable(0, trainable=False)

        self.saver = tf.train.Saver(tf.global_variables())

        parameters = tf.trainable_variables()
        for param in parameters:
            print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                              param.op.device))

    def create_inference_model(self, mode):
        encoder_output, encoder_state = self.create_encoder_model(mode=mode)
        self.encoder_output = encoder_output
        sample_id, final_context_state = self.create_decoder_model(mode=mode, encoder_output=encoder_output,
                                                                   encoder_state=encoder_state)

        return sample_id, final_context_state

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

        tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(SOS)), tf.int32)
        tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(EOS)), tf.int32)

        max_encoder_length = tf.reduce_max(self.src_seq_len)
        max_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * tf.constant(2.0)))
        self.max_encoder_length = max_encoder_length
        self.max_iterations = max_iterations

        if not use_attention:
            sample_id, final_context_state = \
                build_original_inference_beamsearch_decoder(cell=decoder_cell, encoder_output=encoder_output,
                                                            encoder_state=encoder_state,
                                                            batch_size=self.batch_size,
                                                            embedding_decoder=self.embedding_decoder,
                                                            tgt_sos_id=tgt_sos_id, tgt_eos_id=tgt_eos_id,
                                                            max_iterations=max_iterations,
                                                            output_layer=self.output_layer)
        else:
            sample_id, final_context_state = \
                build_attention_inference_beamsearch_decoder(cell=decoder_cell, encoder_output=encoder_output,
                                                             encoder_state=encoder_state, batch_size=self.batch_size,
                                                             embedding_decoder=self.embedding_decoder,
                                                             tgt_sos_id=tgt_sos_id, tgt_eos_id=tgt_eos_id,
                                                             max_iterations=max_iterations,
                                                             source_sequence_length=self.src_seq_len,
                                                             output_layer=self.output_layer, num_unit=num_unit,
                                                             beam_width=self.beam_width)
        return sample_id, final_context_state

    def infer(self, infer_session):
        return infer_session.run([self.sample_id, self.sample_word])

    def decode(self, infer_session):
        sample_id, sample_word = self.infer(infer_session=infer_session)
        sample_word = sample_word.transpose()
        sample_id = sample_id.transpose()

        return sample_word, sample_id


if __name__ == "__main__":

    src_dataset_name = "tst2013.en"
    src_vocab_name = "vocab.en"
    tgt_vocab_name = "vocab.vi"

    src_dataset_path = os.path.join(BASE, src_dataset_name)
    src_vocab_path = os.path.join(BASE, src_vocab_name)
    tgt_vocab_path = os.path.join(BASE, tgt_vocab_name)
    batch_size = 32
    num_buckets = 5

    iterator = get_inference_iterator(src_dataset_path=src_dataset_path, src_vocab_path=src_vocab_path,
                                      batch_size=batch_size)

    infer_model = InferModel(iterator=iterator, tgt_vocab_path=tgt_vocab_path)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    session.run(iterator.initializer)
    sample_word = infer_model.decode(infer_session=session)
    decode_result = session.run(sample_word)
    print(decode_result[0].shape)
    # print(session.run([infer_model.src_seq_len, infer_model.max_iterations, infer_model.max_encoder_length]))
    # print(session.run(infer_model.batch_size))
    # sentence = infer_model.infer(infer_session=session)[1][:, :, 0][0]
    # sentence = list(sentence)
    # sentence = b' '.join(sentence)
    # sentence = sentence.decode("utf-8")
    # print(sentence)
    # print(session.run(infer_model.src_ids))