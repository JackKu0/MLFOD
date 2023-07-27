import tensorflow as tf
import tensorflow.contrib.keras
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras.python.keras.layers import Conv2D,BatchNormalization,Activation,LeakyReLU

slim = tf.contrib.slim

def add_conv(input, out_ch,ksize,stride,leaky=True):
    conv = tf.layers.conv2d(input,filters=out_ch,kernel_size=ksize,strides=stride,padding='same',use_bias=False)
    bn = tf.layers.batch_normalization(conv, trainable=None)
    act = LeakyReLU(0.1) 
    return act(bn)
    
def afm(x_level_0, x_level_1, x_level_2):
    # 定义函数和参数
    #compress_level_0 = add_conv(32, 1, 1)
    #compress_level_1 = add_conv(32, 1, 1)
    #expand = add_conv(32, 3, 1)
    compress_c = 16
    #weight_level_0 = add_conv(compress_c, 1, 1)
    #weight_level_1 = add_conv(compress_c, 1, 1)
    #weight_level_2 = add_conv(compress_c, 1, 1)
    #weight_levels = Conv2D(3,kernel_size=1,strides=(1,1),padding='valid')
    
    # 同维化
    level_0_compressed = add_conv(x_level_0, 32, 1, 1)
    _,H,W,_ = level_0_compressed.shape
    level_0_resized = tf.image.resize_images(level_0_compressed,[360,1200],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    level_1_compressed = add_conv(x_level_1, 32, 1, 1)
    _,H,W,_ = x_level_1.shape
    level_1_resized = tf.image.resize_images(level_1_compressed,[360,1200],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    level_2_resized = x_level_2
    
    # 计算权重
    level_0_weight_v = add_conv(level_0_resized, 16, 1, 1)
    level_1_weight_v = add_conv(level_1_resized, 16, 1 ,1)
    level_2_weight_v = add_conv(level_2_resized, 16, 1, 1)
    
    levels_weight_v = tf.concat((level_0_weight_v, level_1_weight_v, level_2_weight_v),3) # tensorflow BHWC
    # levels_weight = F.softmax(levels_weight, dim=1)
    levels_weight = tf.layers.conv2d(levels_weight_v, filters=3,kernel_size=1,strides=(1,1),padding='valid')
    levels_weight = tf.nn.softmax(levels_weight,-1)
    
    fused_out_reduced = level_0_resized * levels_weight[:,:,:,0:1]+\
                            level_1_resized * levels_weight[:,:,:,1:2]+\
                            level_2_resized * levels_weight[:,:,:,2:]
        
    out = add_conv(fused_out_reduced, 32, 3, 1)
    return out
  
  
""" 
class add_conv(tf.contrib.keras.layers.Layer):
    
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        out: Sequential layers composing a convolution block.
    
    def __init__(self, out_ch,ksize,stride,leaky=True):
        super(add_conv,self).__init__()

        self.conv = Conv2D(filters=out_ch,kernel_size=ksize,strides=stride,padding='same',use_bias=False)
        self.bn = BatchNormalization()
        self.act = LeakyReLU(0.1) 
        #if leaky==True else ReLU(6.0)
    
    def call(self,x):
        return self.act(self.bn(self.conv(x)))
"""
"""


