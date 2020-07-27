import os
import tensorflow as tf
import src.transform as transform
import tfcoreml as tf_converter
import coremltools
from argparse import ArgumentParser
from coremltools.models.neural_network import flexible_shape_utils
from coremltools.models.neural_network.quantization_utils import *

batch_size = 1


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)

    parser.add_argument('--file-name', type=str,
                        dest='file_name', help='model file name',
                        metavar='FILENAME', required=True)

    parser.add_argument('--batch-size', type=str,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=batch_size)

    return parser


def convert_multiarray_output_to_image(spec, feature_name, is_bgr=False):
    for output in spec.description.output:
        if output.name != feature_name:
            continue
        if output.type.WhichOneof('Type') != 'multiArrayType':
            raise ValueError("%s is not a multiarray type" % output.name)
        array_shape = tuple(output.type.multiArrayType.shape)
        channels, height, width = array_shape
        from coremltools.proto import FeatureTypes_pb2 as ft
        if channels == 1:
            output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
        elif channels == 3:
            if is_bgr:
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('BGR')
            else:
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('RGB')
        else:
            raise ValueError("Channel Value %d not supported for image inputs" % channels)
        output.type.imageType.width = width
        output.type.imageType.height = height


def convert_flexible_coremodel(model_path, input_name, output_name):
    spec = coremltools.utils.load_spec(model_path)
    img_size_ranges = flexible_shape_utils.NeuralNetworkImageSizeRange()
    img_size_ranges.add_height_range((100, 1920))
    img_size_ranges.add_width_range((100, 1920))
    flexible_shape_utils.update_image_size_range(spec, feature_name=input_name, size_range=img_size_ranges)
    flexible_shape_utils.update_image_size_range(spec, feature_name=output_name, size_range=img_size_ranges)
    coremltools.utils.save_spec(spec, model_path)

    model_spec = coremltools.utils.load_spec(model_path)
    model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
    coremltools.utils.save_spec(model_fp16_spec, model_path)


def main():
    parser = build_parser()
    options = parser.parse_args()
    ckpt_dir = options.checkpoint_dir
    filename = options.file_name
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), tf.device('/cpu:0'), tf.Session(config=soft_config) as sess:
        # batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 3),
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(ckpt_dir):
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['add_37'])
                with open(ckpt_dir + '/' + filename + '.pb', 'wb') as f:
                    f.write(frozen_graph_def.SerializeToString())
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, ckpt_dir)
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['add_37'])
            with open(ckpt_dir + '/' + filename + '.pb', 'wb') as f:
                f.write(frozen_graph_def.SerializeToString())

    tf_converter.convert(tf_model_path=ckpt_dir + '/' + filename + '.pb',
                         mlmodel_path=ckpt_dir + '/' + filename + '.mlmodel',
                         output_feature_names=['add_37:0'],
                         image_input_names=['img_placeholder__0'])
    model = coremltools.models.MLModel(ckpt_dir + '/' + filename + '.mlmodel')
    # lin_quant_model = quantize_weights(model, 8, "linear")
    # lin_quant_model.save(ckpt_dir + '/' + filename + '.mlmodel')
    spec = model.get_spec()
    convert_multiarray_output_to_image(spec, 'add_37__0', is_bgr=False)
    new_model = coremltools.models.MLModel(spec)
    new_model.save(ckpt_dir + '/' + filename + '_output.mlmodel')
    convert_flexible_coremodel(ckpt_dir + '/' + filename + '_output.mlmodel', 'img_placeholder__0', 'add_37__0')


if __name__ == '__main__':
    main()