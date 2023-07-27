import tensorflow as tf
from tensorflow.contrib import slim

from avod.core.avod_fc_layers import avod_fc_layer_utils


def build(fc_layers_config,
          input_rois, input_weights,
          num_final_classes, box_rep,
          is_training,
          end_points_collection):

    """Builds basic layers  构建基本层

       Args:
           fc_layers_config: Fully connected layers config object 全连接层的配置对象
           input_rois: List of input roi feature maps  输入感兴趣区域的特征图列表
           input_weights: List of weights for each input e.g. [1.0, 1.0]  每个输入的权重列表
           num_final_classes: Final number of output classes, including
               'Background'  输出类的最终数量，包括Background
           box_rep: Box representation (e.g. 'box_3d', 'box_8c', 'box_4c')  框表示
           is_training: Whether the network is training or evaluating  这个网络是培训还是评估
           end_points_collection: End points collection to add entries to  要向其中添加条目的端点集合

       Returns:
           cls_logits: Output classification logits  输出分类分对数
           offsets: Output offsets 输出偏置量
           angle_vectors: Output angle vectors (or None)  输出角向量(或无)
           end_points: End points dict
       """

    # Parse config
    fusion_method = fc_layers_config.fusion_method
    num_layers = fc_layers_config.num_layers
    layer_sizes = fc_layers_config.layer_sizes
    l2_weight_decay = fc_layers_config.l2_weight_decay
    keep_prob = fc_layers_config.keep_prob

    cls_logits, offsets, angle_vectors = \
        _basic_fc_layers(num_layers=num_layers,
                         layer_sizes=layer_sizes,
                         input_rois=input_rois,
                         input_weights=input_weights,
                         fusion_method=fusion_method,
                         l2_weight_decay=l2_weight_decay,
                         keep_prob=keep_prob,
                         num_final_classes=num_final_classes,
                         box_rep=box_rep,
                         is_training=is_training)

    end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    return cls_logits, offsets, angle_vectors, end_points


def build_output_layers(tensor_in,
                        num_final_classes,
                        box_rep,
                        output):
    """Builds flattened output layers  构建平展输出层

    Args:
        tensor_in: Input tensor
        num_final_classes: Final number of output classes, including
            'Background'
        box_rep: Box representation (e.g. 'box_3d', 'box_8c', 'box_4c')

    Returns:
        Output layers
    """
    layer_out = None

    if output == 'cls':
        # Classification
        layer_out = slim.fully_connected(tensor_in,
                                         num_final_classes,
                                         activation_fn=None,
                                         scope='cls_out')
    elif output == 'off':
        # Offsets
        off_out_size = avod_fc_layer_utils.OFFSETS_OUTPUT_SIZE[box_rep]
        if off_out_size > 0:
            layer_out = slim.fully_connected(tensor_in,
                                             off_out_size,
                                             activation_fn=None,
                                             scope='off_out')
        else:
            layer_out = None

    elif output == 'ang':
        # Angle Unit Vectors
        ang_out_size = avod_fc_layer_utils.ANG_VECS_OUTPUT_SIZE[box_rep]
        if ang_out_size > 0:
            layer_out = slim.fully_connected(tensor_in,
                                             ang_out_size,
                                             activation_fn=None,
                                             scope='ang_out')
        else:
            layer_out = None

    return layer_out


def _basic_fc_layers(num_layers, layer_sizes,
                     input_rois, input_weights, fusion_method,
                     l2_weight_decay, keep_prob,
                     num_final_classes, box_rep,
                     is_training):

    if not num_layers == len(layer_sizes):
        raise ValueError('num_layers does not match length of layer_sizes')

    if l2_weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(l2_weight_decay)
    else:
        weights_regularizer = None

    # Feature fusion  特征融合
    fused_features = avod_fc_layer_utils.feature_fusion(fusion_method,
                                                        input_rois,
                                                        input_weights)
    output_names = ['cls', 'off', 'ang']
    cls_logits = None
    offsets = None
    angles = None

    with slim.arg_scope(
            [slim.fully_connected],
            weights_regularizer=weights_regularizer):
        for output in output_names:
            # Flatten  将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡
            fc_drop = slim.flatten(fused_features,
                                   scope=output + '_flatten')
            for layer_idx in range(num_layers):
                fc_name_idx = 6 + layer_idx

                # Use conv2d instead of fully_connected layers.  使用conv2d而不是全连接层
                fc_layer = slim.fully_connected(fc_drop, layer_sizes[layer_idx],
                                                scope=output + '_fc{}'.format(fc_name_idx))

                fc_drop = slim.dropout(fc_layer,
                                       keep_prob=keep_prob,
                                       is_training=is_training,
                                       scope=output + '_fc{}_drop'.format(fc_name_idx))

                fc_name_idx += 1
            if output == 'cls':
                cls_logits= build_output_layers(fc_drop,
                                                num_final_classes,
                                                box_rep,
                                                output)
            elif output == 'off':
                offsets = build_output_layers(fc_drop,
                                              num_final_classes,
                                              box_rep,
                                              output)
            elif output == 'ang':
                angles = build_output_layers(fc_drop,
                                             num_final_classes,
                                             box_rep,
                                             output)

    return cls_logits, offsets, angles
