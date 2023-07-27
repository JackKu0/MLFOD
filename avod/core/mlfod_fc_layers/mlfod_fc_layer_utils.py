import tensorflow as tf

OFFSETS_OUTPUT_SIZE = {
    'box_3d': 6,
    'box_8c': 24,
    'box_8co': 24,
    'box_4c': 10,
    'box_4ca': 10,
}

ANG_VECS_OUTPUT_SIZE = {
    'box_3d': 2,
    'box_8c': 0,
    'box_8co': 0,
    'box_4c': 0,
    'box_4ca': 2,
}


def feature_fusion(fusion_method, inputs, input_weights):
    """Applies feature fusion to multiple inputs  将特征融合应用于多个输入

    Args:
        fusion_method: 'mean' or 'concat'  融合方式：mean或者concat
        inputs: Input tensors of shape (batch_size, width, height, depth)
            If fusion_method is 'mean', inputs must have same dimensions.  如果fusion_method是'mean'，输入必须有相同的尺寸。
            If fusion_method is 'concat', width and height must be the same.  如果fusion_method为'concat'，则宽度和高度必须相同。
        input_weights: Weight of each input if using 'mean' fusion method

    Returns:
        fused_features: Features after fusion
    """

    # Feature map fusion  特征图的融合
    with tf.variable_scope('fusion'):
        fused_features = None

        if fusion_method == 'mean':
            rois_sum = tf.reduce_sum(inputs, axis=0)                       #计算张量tensor沿着某一维度的和，默认在求和后降维。
            rois_mean = tf.divide(rois_sum, tf.reduce_sum(input_weights))
            fused_features = rois_mean

        elif fusion_method == 'concat':
            # Concatenate along last axis  沿最后一轴连接
            last_axis = len(inputs[0].get_shape()) - 1
            fused_features = tf.concat(inputs, axis=last_axis)

        elif fusion_method == 'max':
            fused_features = tf.maximum(inputs[0], inputs[1])

        else:
            raise ValueError('Invalid fusion method', fusion_method)

    return fused_features
