import tensorflow as tf


def build_rnn_cell(mode, unit_type="lstm", num_unit=512, num_layers=2, forget_bias=1.0, dropout=0.2):
    """
    build a rnn cell for encoder and decoder
    :param unit_type: the type of the rnn cell .e.g lstm, gru
    :param num_unit: the number of the cell
    :param num_layers: the number of the layers
    :param forget_bias: forget bias only for lstm
    :param dropout: dropout rate, used in the training mode
    :param mode: one of (TRAIN. EVAL, TEST) .e.g tf.contrib.learn.ModeKeys.TRAIN
    :return: return a cell object whose input is current input and last cell state and gives the output and the next state
             you can use like this
             output, new_cell_state = cell(input, old_cell_state)
    """

    cell_list = list()
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
    for index in range(num_layers):
        single_cell = build_single_cell(unit_type=unit_type, num_unit=num_unit, forget_bias=forget_bias, dropout=dropout)
        cell_list.append(single_cell)

    if len(cell_list) == 1:
        return cell_list[0]
    else:
        return tf.contrib.rnn.MultiRNNCell(cell_list)


def build_single_cell(unit_type, num_unit, forget_bias, dropout):
    """
    :param unit_type: the type of the rnn cell .e.g lstm, gru
    :param num_unit: the number of the cell
    :param forget_bias: forget bias only for lstm
    :param dropout: dropout rate, used in the training mode
    :return: return a cell object whose input is current input and last cell state and gives the output and the next state
             you can use like this
             output, new_cell_state = cell(input, old_cell_state)
    """
    if unit_type == "lstm":
        single_cell = tf.contrib.rnn.BasicLSTMCell(num_unit, forget_bias=forget_bias)
    elif unit_type == "gru":
        single_cell = tf.contrib.rnn.GRUCell(num_unit)
    else:
        raise ValueError("Unknow unit type %s" % unit_type)

    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))

    return single_cell


