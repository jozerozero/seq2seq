import tensorflow as tf


def build_original_decoder(cell, encoder_output, encoder_state, target_sequence_length, source_sequence_length,
                           target_tokens, embedding_decoder):
    """
    :param cell: the rnn cell of the decoder
    :param encoder_output: every time step output of the encoder, but we don't use it in the original decoder
                           but the shape of encoder_output is [max_time, batch_size, encoder_output_size]
    :param encoder_state: the finally cell state of the encoder, which is used for initializing the decoder
    :param target_sequence_length: the length of the target sentence
    :param source_sequence_length: the length of the source sentence
    :param target_tokens : the tokens of the target sentence
    :param embedding_decoder : the table that turn the target tokens to the target word vectors
    :return:
            outputs : contain the hidden state as well as the simple id
            get the hidden state and the simple_id as follow : outputs.rnn_output, outputs.sample_id
            decoder's hidden state in every time state, whose shape is [time step, batch size, decoder_output_size]
            and the sample_id is useless because I did not add an output_layer at tf.contrib.seq2seq.dynamic_decode
    """

    with tf.variable_scope("decoder") as decoder_scope:
        target_tokens = tf.transpose(target_tokens)
        decoder_embed_inp = tf.nn.embedding_lookup(embedding_decoder, target_tokens)

        """
        helper object is such a object that help to read the data for the decoder, the principle of the helper as follow: 
        initialize() and sample(time, output) are the key function
        
        initialize()
        return (finished, next_inputs), finished is a list whose element is boolean and size equals to batch 
        size, used for judge whether a sentence has input into the decoder ; next_inputs is the first time step of the decoder
        
        sample(time, output) 
        param: time : represent the current time step ; output : represent the output at current time step
        return (finished, next_inputs), finished has the same mean as the finished return by the initialize() ; 
        but next_inputs is the really the next input of the decoder
        """
        helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_inp, sequence_length=target_sequence_length,
                                                 time_major=True)

        """
        BasicDecoder is such a object that descrip one decode step, the principle of the BasicDecoder as follow:
        initialize() and step(time, input, state) are the key function
        
        initialize()
        return (finished, next_inputs, initial_state), inside the initialize() call helper's initialize(), so the first two
        elements of the return is the same as the helper's. And the initial_state is the encoder_state
        
        step(time, input, state) can be considered as a processing of one decode
        param : time, represent the time step ; input, input of the current time step ; state, cell state of last time step
        return (outputs, next_state, next_inputs, finished): helper.sample(time, output) is called inside step(...), 
        therefore next_inputs as the finished is the same as helper's. outputs and the next_state too easy to say. -_-!
        """
        my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, encoder_state)

        outputs, final_context_state, _ = \
            tf.contrib.seq2seq.dynamic_decode(my_decoder, output_time_major=True, swap_memory=True, scope=decoder_scope)

    return outputs.rnn_output, final_context_state


def build_attention_decoder(cell, encoder_output, encoder_state, target_sequence_length, source_sequence_length,
                            target_tokens, embedding_decoder, batch_size, num_units):
    """
        the same as
        build_original_decoder(cell, encoder_output, encoder_state, target_sequence_length, source_sequence_length,
                           target_tokens, embedding_decoder):
    """
    with tf.variable_scope("decoder") as decoder_scope:
        target_tokens = tf.transpose(target_tokens)
        decoder_embed_inp = tf.nn.embedding_lookup(embedding_decoder, target_tokens)

        attention_cell, decoder_initial_state = build_attention_cell(cell=cell, encoder_outputs=encoder_output,
                                                                     encoder_state=encoder_state,
                                                                     source_sequence_length=source_sequence_length,
                                                                     batch_size=batch_size, num_unit=num_units)

        helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_inp, sequence_length=target_sequence_length,
                                                   time_major=True)

        my_decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper, decoder_initial_state)

        outputs, final_context_state, _ = \
            tf.contrib.seq2seq.dynamic_decode(my_decoder, output_time_major=True, swap_memory=True, scope=decoder_scope)

        return outputs.rnn_output, final_context_state


def build_original_inference_beamsearch_decoder(cell, encoder_output, encoder_state, batch_size,
                                                embedding_decoder, tgt_sos_id, tgt_eos_id, max_iterations, output_layer,
                                                beam_width=10, length_penalty_weight=0.0):
    """

    :param cell: this is the simple rnn cell
    :param encoder_output: every time step output of the encoder, but we don't use it in the original decoder
                           but the shape of encoder_output is [max_time, batch_size, encoder_output_size]
    :param encoder_state: the finally cell state of the encoder, which is used for initializing the decoder
    :param batch_size: batch_size is used for building a start_tokens with tgt_sos_id
    :param embedding_decoder: use to turn the target word into word vectors
    :param tgt_sos_id: target start of sentence token id
    :param tgt_eos_id: target end of sentence token id
    :param max_iterations : the max iteration time
    :param beam_width : beam search width
    :param length_penalty_weight :Length penalty for beam search. but I don't know what it is ,hehehe~~
    :return:
    """
    '''
         after executing the tile_batch(encoder_state, multiplier), 
         the encoder_state change from [batch_size, output] to [batch_size * multiplier, output]
         for example:
         encoder_state = [[1, 2, 3], [4, 5, 6]]
         new_encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=2)
         new_encoder_state = [[1,2,3],[1,2,3],[4,5,6],[4,5,6]]
    '''
    with tf.variable_scope("decoder") as decoder_scope:
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)

        start_tokens = tf.fill(dims=[batch_size], value=tgt_sos_id)
        end_token = tgt_eos_id
        if beam_width <= 0 or beam_width is None:
            raise ValueError("beam width must be greater than zero")

        my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell, embedding=embedding_decoder, start_tokens=start_tokens,
                                                          end_token=end_token, initial_state=decoder_initial_state,
                                                          beam_width=beam_width, output_layer=output_layer,
                                                          length_penalty_weight=length_penalty_weight)

        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, maximum_iterations=max_iterations,
                                                                            output_time_major=True,
                                                                            swap_memory=True, scope=decoder_scope)
        # logits = tf.no_op() #logits is useless when the mode is Infer
        sample_id = outputs.predicted_ids

        return sample_id, final_context_state


def build_attention_inference_beamsearch_decoder(cell, encoder_output, encoder_state, batch_size,
                                                 embedding_decoder, tgt_sos_id, tgt_eos_id, max_iterations,
                                                 source_sequence_length, output_layer, num_unit,
                                                 beam_width=10, length_penalty_weight=0.0):
    with tf.variable_scope("decoder") as decoder_scope:
        start_tokens = tf.fill(dims=[batch_size], value=tgt_sos_id)
        end_token = tgt_eos_id
        if beam_width <= 0 or beam_width is None:
            raise ValueError("beam width must be greater than zero")

        attention_cell, decoder_initial_state = \
            build_attention_inference_cell(cell=cell, encoder_outputs=encoder_output, encoder_state=encoder_state,
                                           source_sequence_length=source_sequence_length, batch_size=batch_size,
                                           num_unit=num_unit, beam_width=beam_width)

        my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=attention_cell, embedding=embedding_decoder,
                                                          start_tokens=start_tokens, end_token=end_token,
                                                          initial_state=decoder_initial_state, beam_width=beam_width,
                                                          output_layer=output_layer,
                                                          length_penalty_weight=length_penalty_weight)

        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, maximum_iterations=max_iterations,
                                                                            output_time_major=True,
                                                                            swap_memory=True, scope=decoder_scope)
        sample_id = outputs.predicted_ids
        return sample_id, final_context_state


def build_attention_cell(cell, encoder_outputs, encoder_state, source_sequence_length, batch_size, num_unit,
                         attention_option="scaled_luong"):
    """

    :param cell: simple rnn cell
    :param encoder_outputs: encoder's output of each time step, token as memory
    :param encoder_state: the encoder's final time step state, used to initial the decoder
    :param source_sequence_length: the length of source sentence
    :param batch_size: bu jie xi
    :param num_unit: the shape of the memory_layer, .e.g memory_laayer(memory)
    :param attention_option: attention mechanism type .e.g "scaled_luong"
    :return:
            attention_cell : a simple cell wrapped with attention mechanism
            decoder_initial_state
    """
    '''memory is a parameter of attention mechanism, 
    and it's shape must be [batch_size, max_time, endoder_output_size], 
    however the shape of encoder_outputs must be [max_time, batch_size, encoder_output_size], 
    so it needs to be transpose'''
    memory = tf.transpose(encoder_outputs, [1, 0, 2])

    if attention_option == "scaled_luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=num_unit, memory=memory,
                                                                memory_sequence_length=source_sequence_length,
                                                                scale=True)
    else:
        raise ValueError("Unknow attention mechanism %s" % attention_option)

    """attention_layer_size : 
       because the after the attention mechanism, the output should be [decoder: context_vector],
       however it will make the shape larger, so we provide attention_layer_size, a full connection netword will be 
       construct, and restrict the output of the attention_cell, and attention_layer_size does not necessarilly equal
       to num_unit

       It's worth noting that the state of AttentionWarpper is different from the state of simple cell, 
       the state of cell is a Tensor, but the state of AttentionWarpper is a key-value Object, it's key as follow:
       cell_state: beacuse the AttentionWarpper contains a simple rnn cell, cell_state is simpel cell state
       time : the decode time step
       attention : if attention_layer_size is provided, attention is the output of the attention_cell,
                   else attention is the context vector
       alignments : the normalized weights between the memory(encoder outputs) and decoder output
       alignments_history : a TensorArray that record the alignment of all the decoder time step"""
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(cell=cell, attention_mechanism=attention_mechanism,
                                                         attention_layer_size=num_unit)

    decoder_initial_state = attention_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

    return attention_cell, decoder_initial_state


def build_attention_inference_cell(cell, encoder_outputs, encoder_state, source_sequence_length, batch_size, num_unit,
                                   beam_width, attention_option="scaled_luong"):
    memory = tf.transpose(encoder_outputs, [1, 0, 2])
    memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
    source_sequence_length = tf.contrib.seq2seq.tile_batch(source_sequence_length, multiplier=beam_width)
    encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
    batch_size = batch_size * beam_width

    print("attention option", attention_option)
    # exit()
    if attention_option == "scaled_luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=num_unit, memory=memory,
                                                                memory_sequence_length=source_sequence_length,
                                                                scale=True)
    else:
        raise ValueError("Unknow attention mechanism %s" % attention_option)

    attention_cell = tf.contrib.seq2seq.AttentionWrapper(cell=cell, attention_mechanism=attention_mechanism,
                                                         attention_layer_size=num_unit)

    decoder_initial_state = attention_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

    return attention_cell, decoder_initial_state
