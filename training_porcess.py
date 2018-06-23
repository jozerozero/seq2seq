from __future__ import print_function

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import math
from utils.iterator_utils import *
from train_model import TrainingModel
from infer_model import InferModel
from eval_model import EvalModel
from decoder.decode_sentence import decode
from evalution.bleu_eval import evaluate_bleu
import tensorflow as tf
import numpy as np


OUTPUT_PATH = "model_result"
BASE = "model_data/iwslt15"


def create_train_model(batch_size, num_buckets, src_dataset_file="train.en", tgt_dataset_file="train.vi",
                       src_vocab_file="vocab.en", tgt_vocab_file="vocab.vi"):

    src_dataset_path = os.path.join(BASE, src_dataset_file)
    tgt_dataset_path = os.path.join(BASE, tgt_dataset_file)
    src_vocab_path = os.path.join(BASE, src_vocab_file)
    tgt_vocab_path = os.path.join(BASE, tgt_vocab_file)

    training_graph = tf.Graph()

    with training_graph.as_default():
        training_iterator = get_iterator(src_dataset_path=src_dataset_path, tgt_dataset_path=tgt_dataset_path,
                                         src_vocab_path=src_vocab_path, tgt_vocab_path=tgt_vocab_path,
                                         batch_size=batch_size, num_buckets=num_buckets, is_shuffle=False)
        training_model = TrainingModel(iterator=training_iterator)

    return training_graph, training_model


def create_eval_model(batch_size, num_buckets,src_dataset_file="tst2012.en", tgt_dataset_file="tst2012.vi",
                       src_vocab_file="vocab.en", tgt_vocab_file="vocab.vi"):

    src_dataset_path = os.path.join(BASE, src_dataset_file)
    tgt_dataset_path = os.path.join(BASE, tgt_dataset_file)
    src_vocab_path = os.path.join(BASE, src_vocab_file)
    tgt_vocab_path = os.path.join(BASE, tgt_vocab_file)

    eval_graph = tf.Graph()

    with eval_graph.as_default():
        eval_iterator = get_iterator(src_dataset_path=src_dataset_path, tgt_dataset_path=tgt_dataset_path,
                                    src_vocab_path=src_vocab_path, tgt_vocab_path=tgt_vocab_path,
                                    batch_size=1, num_buckets=num_buckets, is_shuffle=False,
                                    src_max_len=None, tgt_max_len=None)

        eval_model = EvalModel(iterator=eval_iterator)

    return eval_graph, eval_model


def create_infer_model(infer_batch_size, src_dataset_file="tst2013.en", src_vocab_name="vocab.en",
                       tgt_vocab_name="vocab.vi"):

    src_dataset_path = os.path.join(BASE, src_dataset_file)
    src_vocab_path = os.path.join(BASE, src_vocab_name)
    tgt_vocab_path = os.path.join(BASE, tgt_vocab_name)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_iterator = get_inference_iterator(src_dataset_path=src_dataset_path, src_vocab_path=src_vocab_path,
                                                batch_size=1)

        infer_model = InferModel(iterator=infer_iterator, tgt_vocab_path=tgt_vocab_path)

    return infer_graph, infer_model


def run_internal_eval(eval_model, eval_session, eval_graph,  model_dir):
    with eval_graph.as_default():
        # load_eval_model, global_steps = create_or_load_model(model=eval_model, model_dir=model_dir,
        #                                                      session=eval_session)
        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        eval_model.saver.restore(eval_session, latest_ckpt)
        eval_session.run(tf.tables_initializer())
        eval_session.run(eval_model.iterator.initializer)

        # print("had load the model")
        # src_tokens, src_size, tgt_in_tokens, tgt_size, logits, tgt_out_tokens, loss = \
        #     eval_session.run([eval_model.src_ids, eval_model.src_seq_len,
        #                       eval_model.tgt_input_ids, eval_model.tgt_seq_len, eval_model.logits,
        #                       eval_model.tgt_output_ids, eval_model.eval_loss])

        total_loss = 0
        total_predict_count = 0
        while True:
            try:
                eval_loss, predict_count, batch_size = eval_model.eval(eval_session)
                total_loss += eval_loss * batch_size
                total_predict_count += predict_count
            except tf.errors.OutOfRangeError:
                print("this is the evalution step, and the dev dataset has been read out")
                break

        tmp = total_loss / total_predict_count
        try:
            perplexity = math.exp(tmp)
        except OverflowError:
            perplexity = float("inf")

    return perplexity


def run_external_eval(infer_model, infer_session, infer_graph, model_dir, label="test",
                      refer_file="model_data/iwslt15/tst2013.vi"):
    with infer_graph.as_default():
        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        infer_model.saver.restore(infer_session, latest_ckpt)
        infer_session.run(tf.tables_initializer())
        infer_session.run(infer_model.iterator.initializer)

    decode_file = os.path.join(model_dir, "output_%s" % label)
    print("decode file : %s" % decode_file)

    decode(model=infer_model, session=infer_session, translated_file=decode_file)

    # decode_file = "/home/lizijian/result/vanilla/output_test"
    score = evaluate_bleu(target_file=decode_file, reference_file=refer_file)
    return score


def train(output_dir, num_train_step=15000, steps_per_stats=200, steps_per_bleu_eval=1000,
          training_batch_size=128, num_bucket=5, infer_batch_size=32):
    print("create training model\n\n")
    training_graph, training_model = create_train_model(batch_size=training_batch_size, num_buckets=num_bucket)
    print("create inference model\n\n")
    infer_graph, infer_model = create_infer_model(infer_batch_size=infer_batch_size)
    print("create evalution model")
    eval_graph, eval_model = create_eval_model(batch_size=training_batch_size, num_buckets=num_bucket)

    summary_name = "training_log"

    log_file = os.path.join(output_dir, "log_%d" % time.time())
    log_f = tf.gfile.GFile(log_file, mode='a')

    config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True

    train_session = tf.Session(graph=training_graph, config=config_proto)
    eval_session = tf.Session(graph=eval_graph, config=config_proto)
    infer_session = tf.Session(graph=infer_graph, config=config_proto)

    with training_graph.as_default():
        latest_ckpt = tf.train.latest_checkpoint(output_dir)
        if latest_ckpt:
            training_model.saver.restore(train_session, latest_ckpt)
            train_session.run(tf.tables_initializer())
        else:
            train_session.run(tf.global_variables_initializer())
            train_session.run(tf.tables_initializer())

        global_steps = training_model.global_step.eval(train_session)

    print("current global step : ", global_steps)

    summary_writer = tf.summary.FileWriter(os.path.join(output_dir, summary_name), training_graph)

    #before training, the iterator must be initialized
    train_session.run(training_model.iterator.initializer)

    checkpoint_loss = 0.0
    checkpoint_predict_count = 0.0

    while global_steps < num_train_step:
        try:
            _, training_loss, predict_count, global_steps, word_count, batch_size = \
                training_model.train(train_session)

            checkpoint_loss += (batch_size * training_loss)
            checkpoint_predict_count += predict_count

            if global_steps % steps_per_stats == 0:
                # print(global_steps)
                train_ppl = math.exp(checkpoint_loss / checkpoint_predict_count)
                print("global step : %d   ppl : %.2f learning rate : %g" %
                      (global_steps, train_ppl, train_session.run(training_model.learning_rate)))
                checkpoint_loss, checkpoint_predict_count = 0.0, 0.0

            if global_steps % steps_per_bleu_eval == 0:
                print("begin saving the model")
                training_model.saver.save(train_session, os.path.join(output_dir, "translate.ckpt"),
                                          global_step=global_steps)
                print("had saved the model")

                print("begin computing perplexity")
                perplexity = run_internal_eval(eval_model=eval_model, eval_session=eval_session, eval_graph=eval_graph,
                                               model_dir=output_dir)
                print("perplexity: ", perplexity)

                print("begin computing bleu")
                bleu_score = run_external_eval(infer_model=infer_model, infer_session=infer_session,
                                               infer_graph=infer_graph,
                                               model_dir=output_dir)
                print("bleu score : ", bleu_score)

        except tf.errors.OutOfRangeError:
            train_session.run(training_model.iterator.initializer)
            continue

    # run_internal_eval(eval_model=eval_model, eval_session=eval_session, eval_graph=eval_graph,
    #                   model_dir=output_dir)
    # print("perplexity: ", perplexity)

    # print("begin computing bleu")
    bleu_score = run_external_eval(infer_model=infer_model, infer_session=infer_session,
                                       infer_graph=infer_graph,
                                       model_dir=output_dir)
    print("bleu score : ", bleu_score)


if __name__ == "__main__":
    # output_file = sys.argv[1]
    output_file = "test"
    output_dir = os.path.join(OUTPUT_PATH, output_file)

    batch_size = 128
    num_buckets = 5
    train(output_dir=output_dir)

    pass
