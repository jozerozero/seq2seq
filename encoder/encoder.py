import tensorflow as tf
import copy


def build_uni_encoder(num_layer, source_tokens, source_length, cell, embedding_encoder):
    """
    build a unidirectional encoder with specified rnn cell .e.g lstm, gru
    :param num_layer: the number of layer of the encoder
    :param source_tokens: tokens index of source sentence, which is the input of encoder
    :param cell: the cell object of the encoder
    :param embedding_encoder: this is a table that map the token index to the word vector
    :return:
            encoder_outputs: the output of encoder of every time step
            encoder_state: the cell state of the last time step, which is used to initial the decoder
    """

    '''the original data is batch major which is [batch_size, time_step, token_id]
    but the input of the rnn is usually the time major which is [time_step, batch_size, token_id]'''
    source_tokens = tf.transpose(source_tokens)

    with tf.variable_scope("encode") as scope:
        dtype = scope.dtype
        '''this is input of the encoder whose shape is [time_step, batch_size, embedding_size]'''
        encoder_embed_input = tf.nn.embedding_lookup(embedding_encoder, source_tokens)

        '''return the output of the decoder of every time step and the finall cell state'''
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell=cell, inputs=encoder_embed_input, dtype=dtype,
                          sequence_length=source_length, time_major=True)

    return encoder_output, encoder_state


def build_bi_encoder(num_layer, source_tokens, source_length, cell, embedding_encoder):
    """
    build a unidirectional encoder with specified rnn cell .e.g lstm, gru
    :param num_layer: the number of layer of the encoder
    :param source_tokens: tokens index of source sentence, which is the input of encoder
    :param cell: the cell object of the encoder
    :param embedding_encoder: this is a table that map the token index to the word vector
    :return:
            encoder_outputs: the output of encoder of every time step
            encoder_state: the cell state of the last time step, which is used to initial the decoder
    """
    source_tokens = tf.transpose(source_tokens)

    with tf.variable_scope("encode") as scope:
        dtype = scope.dtype
        '''this is input of the encoder whose shape is [time_step, batch_size, embedding_size]'''
        encoder_embed_input = tf.nn.embedding_lookup(embedding_encoder, source_tokens)
        fw_cell = cell
        bw_cell = copy.deepcopy(cell)
        '''return the output of the decoder of every time step and the finall cell state'''
        bi_output, bi_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=encoder_embed_input,
                                                              dtype=dtype, sequence_length=source_length,
                                                              time_major=True)
        num_bi_layers = int(num_layer / 2)
        encoder_output = tf.concat(bi_output, axis=-1)

        if num_bi_layers == 1:
            # print("bi_state is equal to encoder_state")
            encoder_state = bi_state
        else:
            encoder_state = list()
            for layer_id in range(num_bi_layers):
                encoder_state.append(bi_state[0][layer_id])
                encoder_state.append(bi_state[1][layer_id])
            encoder_state = tuple(encoder_state)

    return encoder_output, encoder_state
