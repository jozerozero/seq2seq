from __future__ import print_function
import tensorflow as tf


def decode(model, session, translated_file, beam_width=10, tgt_eos="</s>"):

    num_sentence = 0
    with tf.gfile.GFile(translated_file, mode="w") as result_file:
        result_file.write("")

        while True:
            if model.mode != tf.contrib.learn.ModeKeys.INFER:
                raise ValueError("model should be in the inference mode, please check it")

            try:
                sample_word, sample_id = model.decode(session)
                # sample_word, sample_id, src_token, src_size, encoder_output = \
                #     session.run([model.sample_word, model.sample_id, model.src_ids,
                #                                                  model.src_seq_len, model.encoder_output])
                # sample_id = sample_id[:, :, 0]
                # sample_id = sample_id.reshape([-1])
                # print(encoder_output)
                # print(sample_id.tolist())
                # print(src_token.tolist())
                # print(src_size)
                # print(sample_word[:, :, 0])
                # print(sample_word)
                # print(sample_id.shape)
                # exit()
                # print(sample_word)
                if beam_width > 0:
                    sample_word = sample_word[0]

                num_sentence += len(sample_word)
                for sentence_id in range(len(sample_word)):
                    """cut the sample word when there is a eos symbol"""
                    sentence = sample_word[sentence_id, :].tolist()
                    if tgt_eos in sentence:
                        sentence = sentence[: sentence.index(tgt_eos)]

                    translation_sentence = b' '.join(sentence)
                    result_file.write("%s\n" % translation_sentence)

            except tf.errors.OutOfRangeError:
                print("done, test dataset had been decode, the number of the sentences is %d" % num_sentence)
                break

