from __future__ import print_function
from tensorflow.python.ops import lookup_ops
import tensorflow as tf
import os


UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0


def create_vocab_table(src_vocab_path, tgt_vocab_path):

    src_vocab_table = lookup_ops.index_table_from_file(src_vocab_path, default_value=UNK_ID)
    tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_path, default_value=UNK_ID)
    return src_vocab_table, tgt_vocab_table


def create_src_vocab_table(src_vocab_path):
    src_vocab_table = lookup_ops.index_table_from_file(src_vocab_path, default_value=UNK_ID)
    return src_vocab_table


if __name__ == "__main__":
    BASE = "/home/lizijian/dataset/en-vi/"
    src_vocab_name = "vocab.en"
    tgt_vocab_name = "vocab.vi"
    src_vocab_path = os.path.join(BASE, src_vocab_name)
    tgt_vocab_path = os.path.join(BASE, tgt_vocab_name)

    create_vocab_table(src_vocab_path=src_vocab_path, tgt_vocab_path=tgt_vocab_path)

    pass