"""Imports a model metagraph and checkpoint file, converts the variables to constants
and exports the model as a graphdef protobuf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import graph_util
import tensorflow as tf
import argparse
import sys
from generate_detections import _network_factory
from tensorflow.contrib import slim


def preprocess_fn(image, is_training=False, enable_more_augmentation=True):
    image = image[:, :, ::-1]  # BGR to RGB
    if is_training:
        image = tf.image.random_flip_left_right(image)
        if enable_more_augmentation:
            image = tf.image.random_brightness(image, max_delta=50)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image


def main(checkpoint_path, pb_path):
    """

    :param checkpoint_path:
    :param pb_path:
    :return:
    """
    image_shape = 128, 64, 3
    loss_mode = 'cosine'
    graph = tf.Graph()

    factory_fn = _network_factory(num_classes=1501, is_training=False, weight_decay=1e-8)

    with graph.as_default() as G1:
        image_var = tf.placeholder(tf.uint8, (None,) + image_shape)

        preprocessed_image_var = tf.map_fn(
            lambda x: preprocess_fn(x, is_training=False),
            tf.cast(image_var, tf.float32))

        l2_normalize = loss_mode == "cosine"
        feature_var, _ = factory_fn(
            preprocessed_image_var, l2_normalize=l2_normalize, reuse=None)
        feature_dim = feature_var.get_shape().as_list()[-1]

        session = tf.Session()
        with session:
            # slim.get_or_create_global_step()
            tf.train.get_or_create_global_step()
            get_variables_torestore = slim.get_variables_to_restore()
        for var in get_variables_torestore:
            print("DEBUG: ", var)

        # restore graph
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
            checkpoint_path, get_variables_torestore)
        session.run(init_assign_op, feed_dict=init_feed_dict)

        # write to pb
        g1_graph_def = G1.as_graph_def()
        output_node_names = "truediv"
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            session,  # The session
            g1_graph_def,  # input_graph_def is useful for retrieving the nodes
            output_node_names.split(",")
        )
        # save the pb
        with tf.gfile.GFile(pb_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', type=str,
                        help='path/to/checkpoint')
    parser.add_argument('--pb_path', type=str,
                        help='output pb path. path/to/pb')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments()
    main(args.checkpoint_path)