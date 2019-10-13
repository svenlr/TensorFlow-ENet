import numpy as np
import tensorflow as tf

from tensorflow.python import pywrap_tensorflow


def get_tensors_in_checkpoint_file(file_name, all_tensors=True, tensor_name=None):
    varlist = []
    var_value = []
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            varlist.append(key)
            var_value.append(reader.get_tensor(key))
    else:
        varlist.append(tensor_name)
        var_value.append(reader.get_tensor(tensor_name))
    return (varlist, var_value)


def match_loaded_and_memory_tensors(loaded_tensors):
    full_var_list = list()
    # Loop all loaded tensors
    for i, (tensor_name, tensor_loaded) in enumerate(zip(loaded_tensors[0], loaded_tensors[1])):
        try:
            # Extract tensor
            tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name + ":0")
            if not np.array_equal(tensor_aux.shape, tensor_loaded.shape) \
                    and not np.array_equal(tensor_aux.shape, tensor_loaded.shape[::-1]):
                print('Weight mismatch for tensor {}: Current model: {}, Checkpoint: {}'.format(tensor_name, tensor_aux.shape,
                                                                                              tensor_loaded.shape))
            else:
                full_var_list.append(tensor_aux)
        except:
            print('Loaded a tensor from checkpoint which has not been found in model: ' + tensor_name)
    return full_var_list


def restore_matching_weights(sess, data_path):
    vars_in_checkpoint = get_tensors_in_checkpoint_file(file_name=data_path)
    loadable_tensors = match_loaded_and_memory_tensors(vars_in_checkpoint)
    loader = tf.train.Saver(var_list=loadable_tensors)
    loader.restore(sess, data_path)


def gen_feed_dict_from_checkpoint(data_path):
    vars_in_checkpoint = get_tensors_in_checkpoint_file(file_name=data_path)
    checkpoint_dict = dict(list(zip(vars_in_checkpoint[0], vars_in_checkpoint[1])))
    feed_dict = {}
    loadable_tensors = match_loaded_and_memory_tensors(vars_in_checkpoint)
    for tensor in loadable_tensors:
        tensor_name = tensor.name.replace(":0", "")
        feed_dict[tensor] = checkpoint_dict[tensor_name]
    return feed_dict
