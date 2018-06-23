from __future__ import print_function

import os

import tensorflow as tf

import vocab_utils

num_thread = 4


def get_inference_iterator(src_dataset_path, src_vocab_path, batch_size=32, source_reverse=True):
    """
    :param src_dataset_path: inference source sentence dataset path
    :param src_vocab_path: inference source vocabulary path
    :param batch_size: inference batch size
    :param source_reverse: whether reverse the order of the source sentence
    :return: return a iterator that contain the source sentence and the length of source sentence
    """

    src_vocab_table = vocab_utils.create_src_vocab_table(src_vocab_path=src_vocab_path)

    # session = tf.Session()
    # session.run(tf.tables_initializer())
    # # session.run(src_vocab_table)
    # print(src_vocab_table)
    # exit()
    src_dataset = tf.contrib.data.TextLineDataset(src_dataset_path)

    src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

    src_dataset = src_dataset.map(lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

    if source_reverse:
        src_dataset = src_dataset.map(lambda src: tf.reverse(src, axis=[0]))

    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(vocab_utils.EOS)), tf.int32)

    def batching_func(x):
        return x.padded_batch(batch_size,
                              padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])),
                              padding_values=(src_eos_id, 0))

    batched_dataset = batching_func(src_dataset)

    iterator = batched_dataset.make_initializable_iterator()
    return iterator


def get_iterator(src_dataset_path, tgt_dataset_path,
                 src_vocab_path, tgt_vocab_path,
                 batch_size, num_buckets, source_reverse=True,
                 is_shuffle=True, src_max_len=50, tgt_max_len=50):
    '''
    :param src_dataset_path: source dataset path
    :param tgt_dataset_path: target dataset path
    :param src_vocab_path: source dataset vocabulary
    :param tgt_vocab_path: target dataset vocabulary
    :param batch_size: batch size
    :param source_reverse: a boolean value, whether to generate the reverse order source input
    :param number_bucket: the number of the bucket

    :return: the return of the method is a iterator obj, you can get the data in the iterator through:
            (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = iterator.get_next()

            src_ids : this is the source sentence token ids
            tgt_input_ids : this is the target sentence token ids for decoders
            tgt_output_ids: this is the target sentence token ids for labels
            scr_seq_len : this is the source sentence length use for dynamic RNN
            tgt_seq_len : this is the target sentence length use for dynamic RNN
    '''
    num_thread = 8  #the number of thread to preprocessing the data
    output_buffer_size=batch_size * 10  # representing the number of elements
                                        # from this dataset from which the new dataset will sample.

    src_dataset = tf.contrib.data.TextLineDataset(src_dataset_path)
    tgt_dataset = tf.contrib.data.TextLineDataset(tgt_dataset_path)

    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_table(src_vocab_path=src_vocab_path,
                                                                      tgt_vocab_path=tgt_vocab_path)

    '''the follow code is about data preprocessing'''
    #combine the source dataset and the target dataset
    src_tgt_dataset = tf.contrib.data.Dataset.zip((src_dataset, tgt_dataset))

    if is_shuffle:
        print("shuffle the dataset")
        src_tgt_dataset = src_tgt_dataset.shuffle(buffer_size=output_buffer_size, seed=0)

    #segment the sentence by the block .e.g ['this is a dog'] ==> ['this', 'is', 'a', 'dog']
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (tf.string_split([src]).values,
                                                            tf.string_split([tgt]).values),
                        num_threads=num_thread, output_buffer_size=output_buffer_size)

    #remove the sentence whose length equals to zero
    def length_filter(src, tgt):
        return tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0)
    src_tgt_dataset = src_tgt_dataset.filter(length_filter)

    #restric the len of the source sentence to src_max_len
    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src[: src_max_len], tgt),
                            num_threads=num_thread, output_buffer_size=output_buffer_size)

    #restric the len of the target sentence to tgt_max_len
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src, tgt[: tgt_max_len]),
                            num_threads=num_thread, output_buffer_size=output_buffer_size)


    if source_reverse:
        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (tf.reverse(src, axis=[0]), tgt),
                        num_threads=num_thread, output_buffer_size=output_buffer_size)

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                                                            tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
                                          num_threads=num_thread, output_buffer_size=output_buffer_size)

    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    # before execute this code : src_tgt_dataset contain(source_sentence, target_sentence)
    # after execute this code : src_tgt_dataset contain(source_sentence, <sos>_tgt_sentence, tgt_sentence_<eos>)
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(vocab_utils.EOS)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(vocab_utils.EOS)), tf.int32)
    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(vocab_utils.SOS)), tf.int32)

    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src,
                                                            tf.concat(([tgt_sos_id], tgt), 0),
                                                            tf.concat((tgt, [tgt_eos_id]), 0)),
                                          num_threads=num_thread, output_buffer_size=output_buffer_size)

    # Add in the word counts.  Subtract one from the target to avoid counting
    # the target_input <eos> tag (resp. target_output <sos> tag).
    # before execute this code src_tgt_dataset contain(source_sentence, <sos>_tgt_sentence, tgt_sentence_<eos>)
    # after execute this code src_tgt_dataset contain
    # (source sentence, <sos>_tgt_sentence, tgt_sentence_<eos>, len(source sentence), len(<sos>_tat_sentence))
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt_in, tgt_out:
                                          (src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
                                          num_threads=num_thread, output_buffer_size=output_buffer_size)
    #
    # return src_tgt_dataset.make_initializable_iterator()
    # exit()
    '''
        as a matter of fact, I can really tell the detail of this code, I just know I group the data with buckets
    '''
    # number_bucket

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([None]),  # tgt_input
                           tf.TensorShape([None]),  # tgt_output
                           tf.TensorShape([]),  # src_len
                           tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(src_eos_id,  # src
                            tgt_eos_id,  # tgt_input
                            tgt_eos_id,  # tgt_output
                            0,  # src_len -- unused
                            0))  # tgt_len -- unused

    def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
        # Calculate bucket_width by maximum source sequence length.
        # Pairs with length [0, bucket_width) go to bucket 0, length
        # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
        # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
        if src_max_len:
            bucket_width = (src_max_len + num_buckets - 1) // num_buckets
        else:
            bucket_width = 10

        # Bucket sentence pairs by the length of their source sentence and target
        # sentence.
        bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.group_by_window(
        key_func=key_func, reduce_func=reduce_func, window_size=batch_size)

    batch_iterator = batched_dataset.make_initializable_iterator()

    return batch_iterator


def get_iterator_verification():
    """
        this method use for unit test
        python iterator_utils.py
        """
    BASE = "/home/lizijian/dataset/en-vi/"
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
                            batch_size=batch_size, num_buckets=num_buckets, is_shuffle=True)

    """
    warning: there is a order of initializing variables and initializing tables
             you should initialize the tables first and then the variables
    """
    session = tf.Session()
    session.run(tf.tables_initializer())
    session.run(tf.global_variables_initializer())

    session.run(iterator.initializer)

    data = iterator.get_next()[0]
    # print(session.run(data))
    for i in range(2000):
        try:
            print(i, session.run(data).shape)
        except tf.errors.OutOfRangeError:
            session.run(iterator.initializer)


def get_inference_iterator_verification():
    BASE = "/home/lizijian/dataset/en-vi/"
    src_dataset_name = "train.en"
    src_vocab_name = "vocab.en"
    src_dataset_name = os.path.join(BASE, src_dataset_name)
    src_vocab_name = os.path.join(BASE, src_vocab_name)

    iterator = get_inference_iterator(src_dataset_path=src_dataset_name, src_vocab_path=src_vocab_name)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    session.run(iterator.initializer)
    data = iterator.get_next()
    print(session.run(data[1]))
    pass


if __name__ == "__main__":
    get_iterator_verification()
    # get_iterator_verification()