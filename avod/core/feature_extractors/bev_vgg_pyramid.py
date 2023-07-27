import tensorflow as tf

from avod.core.feature_extractors import bev_feature_extractor
from avod.core.feature_extractors.AFM import *
slim = tf.contrib.slim

def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
         squeeze = tf.reduce_mean(input_x,[1,2])#对每个通道取全局最大化
         excitation = tf.layers.dense(squeeze, units=out_dim / ratio, name=layer_name+'_fully_connected1')
         excitation = tf.nn.relu(excitation)
         excitation = tf.layers.dense(excitation, units=out_dim, name=layer_name+'_fully_connected2')
         excitation = tf.sigmoid(excitation)
         excitation = tf.reshape(excitation, [-1,1,1,out_dim])
         scale = input_x * excitation
         return scale

class BevVggPyr(bev_feature_extractor.BevFeatureExtractor):
    """Contains modified VGG model definition to extract features from
    Bird's eye view input using pyramid features.
    """

    def vgg_arg_scope(self, weight_decay=0.0005):
        """Defines the VGG arg scope.

        Args:
          weight_decay: The l2 regularization coefficient.

        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(
                                weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def build(self,
              inputs,
              input_pixel_size,
              is_training,
              scope='bev_vgg_pyr'):
        """ Modified VGG for BEV feature extraction with pyramid features

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The last op containing the log predictions and end_points dict.
        """
        vgg_config = self.config

        with slim.arg_scope(self.vgg_arg_scope(
                weight_decay=vgg_config.l2_weight_decay)):
            with tf.variable_scope(scope, 'bev_vgg_pyr', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    # Pad 700 to 704 to allow even divisions for max pooling
                    padded = tf.pad(inputs, [[0, 0], [4, 0], [0, 0], [0, 0]])  #[1,704,800,6]

                    # Encoder
                    conv1 = slim.repeat(padded,
                                        vgg_config.vgg_conv1[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv1[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv1')      #[1,704,800,32]
                    conv1=Squeeze_excitation_layer(input_x=conv1,out_dim=32, ratio=4, layer_name='conv1')
                    pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')  # [1,352,400,32]
                    # pool1=Squeeze_excitation_layer(input_x=pool1,out_dim=32, ratio=4, layer_name='pool1')

                    conv2 = slim.repeat(pool1,
                                        vgg_config.vgg_conv2[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv2[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv2')      #[1,352,400,64]
                    # conv2=Squeeze_excitation_layer(input_x=conv2,out_dim=64, ratio=4, layer_name='conv2')
                    pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')  # [1,176,200,64]
                    # pool2=Squeeze_excitation_layer(input_x=pool2,out_dim=64, ratio=4, layer_name='pool2')

                    conv3 = slim.repeat(pool2,
                                        vgg_config.vgg_conv3[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv3[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv3')      #[1,176,200,128]
                    # conv3=Squeeze_excitation_layer(input_x=conv3,out_dim=128, ratio=4, layer_name='conv3')
                    pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')  # [1,88,100,128]
                    # pool3=Squeeze_excitation_layer(input_x=pool3,out_dim=128, ratio=4, layer_name='pool3')

                    conv4 = slim.repeat(pool3,
                                        vgg_config.vgg_conv4[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv4[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv4')       #[1,88,100,256]

                    # Decoder (upsample and fuse features)
                    upconv3 = slim.conv2d_transpose(
                        conv4,
                        vgg_config.vgg_conv3[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv3')              #[1,176,200,128]

                    concat3 = tf.concat(
                        (conv3, upconv3), axis=3, name='concat3')   #[1,176,200,256]
                    pyramid_fusion3 = slim.conv2d(
                        concat3,
                        vgg_config.vgg_conv2[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion3')       #[1,176,200,64]

                    upconv2 = slim.conv2d_transpose(
                        pyramid_fusion3,
                        vgg_config.vgg_conv2[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv2')               #[1,352,400,64]

                    concat2 = tf.concat(
                        (conv2, upconv2), axis=3, name='concat2')   #[1,352,400,128]
                    pyramid_fusion_2 = slim.conv2d(
                        concat2,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion2')        #[1,352,400,32]

                    upconv1 = slim.conv2d_transpose(
                        pyramid_fusion_2,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv1')                #[1,704,800,32]

                    concat1 = tf.concat(
                        (conv1, upconv1), axis=3, name='concat1')   #[1,704,800,64]
                    pyramid_fusion1 = slim.conv2d(
                        concat1,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion1')        #[1,704,800,32]
                        

                    P1_AFM = afm(pyramid_fusion3, pyramid_fusion_2, pyramid_fusion1)
                    pyramid_fusion1 = P1_AFM
                    # Slice off padded area
                    sliced = pyramid_fusion1[:, 4:]

                feature_maps_out = sliced

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return feature_maps_out, end_points
