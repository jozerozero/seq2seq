from __future__ import print_function

import codecs
import re

import tensorflow as tf

import bleu


def evaluate_bleu(target_file, reference_file):
    if tf.gfile.Exists(target_file) and tf.gfile.Exists(reference_file) and target_file != "" and reference_file != "":
        score = _bleu(ref_file=reference_file, trans_file=target_file)
        return score
    raise ValueError("can not find %s or %s" % (target_file, reference_file))


# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, trans_file, bpe_delimiter=None):
  """Compute BLEU scores and handling BPE."""
  max_order = 4
  smooth = False

  ref_files = [ref_file]
  reference_text = []
  for reference_filename in ref_files:
    with codecs.getreader("utf-8")(
        tf.gfile.GFile(reference_filename, "rb")) as fh:
      reference_text.append(fh.readlines())

  per_segment_references = []
  for references in zip(*reference_text):
    reference_list = []
    for reference in references:
      reference = _clean(reference, bpe_delimiter)
      reference_list.append(reference.split(" "))
    per_segment_references.append(reference_list)

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = _clean(line, bpe_delimiter)
      translations.append(line.split(" "))

  # bleu_score, precisions, bp, ratio, translation_length, reference_length
  bleu_score, _, _, _, _, _ = bleu.compute_bleu(
      per_segment_references, translations, max_order, smooth)
  return 100 * bleu_score


def _clean(sentence, bpe_delimiter):
  """Clean and handle BPE delimiter."""
  sentence = sentence.strip()

  # BPE
  if bpe_delimiter:
    sentence = re.sub(bpe_delimiter + " ", "", sentence)

  return sentence


if __name__ == "__main__":
    target_file = "/home/lizijian/result/vanilla_big_batch/output_test"
    reference_file = "/home/lizijian/dataset/en-vi/tst2013.vi"
    bleu_score = evaluate_bleu(target_file=target_file, reference_file=reference_file)
    print(bleu_score)