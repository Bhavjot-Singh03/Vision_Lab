K.clear_session()
from tensorflow.keras.applications import DenseNet121

## Transformer Mechanism

def MLP(x, mlp_dim, dim, dropout_rate = 0.2):

    x = Dense(mlp_dim, activation = 'swish')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dim)(x)
    x = Dropout(dropout_rate)(x)

    return x

def Trans_Encoder(inputs, num_heads, hidden_dim, mlp_dim):
    
    skip_1 = inputs
    x = LayerNormalization()(inputs)
    x = MultiHeadAttention(num_heads = num_heads, key_dim = hidden_dim)(x, x)
    x = Add()([skip_1, x])

    skip_2 = x
    x = LayerNormalization()(x)
    x = MLP(x, mlp_dim, hidden_dim)
    x = Add()([skip_2, x])

    return x

def MHA_RESIDUAL_CONV(inputs, dilation_rate = 1, filters = 256):
    
    B, H, W, C = inputs.shape
    skip = inputs
    
    x = Conv2D(filters = filters, kernel_size = (1,1), dilation_rate = dilation_rate, padding = 'same', use_bias = False)(inputs)
    x = BatchNormalization()(x)
    
    """Patch Embeddings"""
    patch_embed = Activation('swish')(x)
    _, h, w, f = patch_embed.shape
    patch_embed = Reshape((h*w, f))(patch_embed) # (64, 256)
    
    """Positional Embedding -> Number of patches : 128*128/16*16 = 64"""
    positions = tf.range(start = 0, limit = 64, delta = 1) # (64, )
    pos_embed = Embedding(input_dim = 64, output_dim = filters)(positions) # (64, 256)
    
    embedding = patch_embed + pos_embed
    x = embedding

    T = Trans_Encoder(x, 9, filters, filters * 2)
    T = LayerNormalization()(T)
    T = Reshape((H,W,filters))(T)
   
    skip = Conv2D(filters = filters, kernel_size = (1,1), dilation_rate = dilation_rate, padding = 'same', use_bias = False)(skip)
    skip = Add()([T, skip])
    skip = BatchNormalization()(skip)
    skip = Activation('swish')(skip)
        
    return skip


## ASPP Module

def ASPP(features):
    shape = features.shape

    a1 = AveragePooling2D(pool_size = (shape[1], shape[2]))(features) 
    a1 = Conv2D(filters = 256, kernel_size = (1,1), padding = 'same', use_bias = False)(a1)
    a1 = BatchNormalization()(a1)
    a1 = Activation('swish')(a1)
    a1 = Conv2DTranspose(filters = 256, kernel_size = (shape[1], shape[2]), strides = (shape[1], shape[2]))(a1) 

    a2 = MHA_RESIDUAL_CONV(features)

    a3 = MHA_RESIDUAL_CONV(features, dilation_rate = 6)
    
    a4 = MHA_RESIDUAL_CONV(features, dilation_rate = 12)

    a5 = MHA_RESIDUAL_CONV(features, dilation_rate = 18)

    a = Concatenate()([a1, a2, a3, a4, a5]) 

    a = MHA_RESIDUAL_CONV(a)

    return a

## DeepLabV3plus instantiation

def DeepLabV3(shape, output_channels, output_activation):
    inputs = Input(shape)

    """Using DenseNet121 as a base_model"""
    base_model = DenseNet121(weights = 'imagenet', include_top = False, input_tensor=inputs)
    features = base_model.get_layer('pool4_conv').output # (None, 8, 8, 512)

    """Atrous Spatial Pyramid Pooling"""
    e = ASPP(features)
    # Upsampling by 4
    e_a = Conv2DTranspose(filters = 256, kernel_size = (2,2), strides = (2,2))(e) # (None, 16, 16, 256)

    """Extracting Low level feature from DenseNet121""" 
    d_l = base_model.get_layer('pool3_conv').output # (None, 16, 16, 128)
    d_l = Conv2D(filters = 64, kernel_size = (1,1), padding = 'same', use_bias = False)(d_l)
    d_l = BatchNormalization()( d_l)
    d_l = Activation('swish')( d_l)

    ed = Concatenate()([e_a, d_l]) 
  
    d = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', use_bias = False)(ed)
    d = BatchNormalization()(d)
    d = Activation('swish')(d)
    
    d = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', use_bias = False)(d)
    d = BatchNormalization()(d)
    d = Activation('swish')(d)
  
    d = Conv2DTranspose(filters = 256, kernel_size = (8,8), strides = (8,8))(d) 

    output = Conv2D(filters = output_channels, kernel_size = (1,1), activation = output_activation)(d)
  
    return Model(inputs, output, name = 'MHA_DEEPLABV3_PLUS')

## Instantiation of the model

K.clear_session()
output_channels = 1
activation = 'sigmoid' 
shape = (128, 128, 3)

MHA_DEEPLAB = DeepLabV3(shape, output_channels, activation)

total_params = 0
for layer in MHA_DEEPLAB.trainable_weights:
    total_params += tf.keras.backend.count_params(layer)

print(f"Total number of parameters: {total_params}")


