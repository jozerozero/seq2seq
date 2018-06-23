from __future__ import print_function

import tensorflow as tf

import os

from iterator_utils import get_iterator, get_inference_iterator


class TestIterator(tf.test.TestCase):

    def test_get_inference_iterator(self):
        base_path = "../model_data/iwslt15/"

        src_dataset_path = os.path.join(base_path, "tst2012.en")
        src_vocab_path = os.path.join(base_path, "vocab.en")

        inference_iterator = \
            get_inference_iterator(src_vocab_path=src_vocab_path, src_dataset_path=src_dataset_path, batch_size=1553,
                                   source_reverse=True)

        with self.test_session() as session:
            session.run(tf.tables_initializer())
            session.run(tf.global_variables_initializer())

            session.run(inference_iterator.initializer)

            src_infer_file = open("../test_result/src_infer.txt", mode="a")
            src_infer_seq_len_file = open("../test_result/src_infer_seq_len.txt", mode="a")

            for i in range(10000):
                try:
                    src, src_seq_len = session.run(inference_iterator.get_next())

                    for src_sen, src_seq_one_len in zip(src, src_seq_len):
                        src_infer_file.write(" ".join([str(n) for n in src_sen]) + "\n")
                        src_infer_seq_len_file.write(str(src_seq_one_len) + "\n")
                        pass

                except tf.errors.OutOfRangeError:
                    break
                pass
            pass

    def test_get_iteartor(self):

        base_path = "../model_data/iwslt15/"

        src_dataset_path = os.path.join(base_path, "train.en")
        tgt_dataset_path = os.path.join(base_path, "train.vi")
        src_vocab_path = os.path.join(base_path, "vocab.en")
        tgt_vocab_path = os.path.join(base_path, "vocab.vi")

        train_iterator = \
            get_iterator(src_dataset_path=src_dataset_path, tgt_dataset_path=tgt_dataset_path,
                         src_vocab_path=src_vocab_path, tgt_vocab_path=tgt_vocab_path, batch_size=128, num_buckets=1,
                         source_reverse=True, is_shuffle=False, src_max_len=50, tgt_max_len=50)

        with self.test_session() as session:
            session.run(tf.tables_initializer())
            session.run(tf.global_variables_initializer())

            session.run(train_iterator.initializer)

            src_file = open("../test_result/src.txt", mode="a")
            tgt_in_file = open("../test_result/tgt_in.txt", mode="a")
            tgt_out_file = open("../test_result/tgt_out.txt", mode="a")
            src_seq_len_file = open("../test_result/src_seq_len_file.txt", mode="a")
            tgt_seq_len_file = open("../test_result/tgt_seq_len_file.txt", mode="a")

            for i in range(10000):
                try:
                    src, tgt_in, tgt_out, src_seq_len, tgt_seq_len = session.run(train_iterator.get_next())
                    for src_sen, tgt_in_sen, tgt_out_sen, src_seq_one_len, tgt_seq_one_len in \
                            zip(src, tgt_in, tgt_out, src_seq_len, tgt_seq_len):

                        src_file.write(" ".join([str(n) for n in src_sen]) + "\n")
                        tgt_in_file.write(" ".join([str(n) for n in tgt_in_sen]) + "\n")
                        tgt_out_file.write(" ".join([str(n) for n in tgt_out_sen]) + "\n")
                        src_seq_len_file.write(str(src_seq_one_len) + "\n")
                        tgt_seq_len_file.write(str(tgt_seq_one_len) + "\n")

                except tf.errors.OutOfRangeError:
                    break


if __name__ == "__main__":

    tf.test.main()