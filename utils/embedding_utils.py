import tensorflow as tf


def create_embedding_layer_for_encoder_and_decoder(src_embedding_size, tgt_embedding_size,
                                                   src_vocab_size, tgt_vocab_size, dtype=tf.float32):

    with tf.variable_scope("embedding_layer", dtype=dtype):
        with tf.variable_scope("encoding"):
            embedding_encoder = tf.get_variable("embedding_encoder", [src_vocab_size, src_embedding_size], dtype=dtype)

        with tf.variable_scope("decoding"):
            embedding_decoder = tf.get_variable("embedding_decoder", [tgt_vocab_size, tgt_embedding_size], dtype=dtype)

    return embedding_encoder, embedding_decoder