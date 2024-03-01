import tensorflow as tf

def build_model():
    base_model = tf.keras.applications.MobileNetV2(  # take mobile net v2 as downsampling model
        input_shape=(768, 768, 3),
        include_top=False,
    )

    layer_names = [
        'block_1_expand_relu',   # 384x384
        'block_3_expand_relu',   # 192x192
        'block_6_expand_relu',   # 96x96
        'block_13_expand_relu',  # 48x48
        'block_16_project',      # 24x24
    ]  # define the skip connection layers
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(
        inputs=base_model.input, outputs=base_model_outputs
    )  # change outputs to the skip connection layers

    up_stack = [
        upscale_layer(512),  # 24x24 -> 48x48
        upscale_layer(256),  # 48x48 -> 96x96
        upscale_layer(128),  # 96x96 -> 192x192
        upscale_layer(64),   # 192x192 -> 384x384
    ]

    inputs = tf.keras.layers.Input(shape=[768, 768, 3])
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    preprocessed_inputs = preprocess_input(inputs)
    
    # downsampling through the model
    skips = down_stack(preprocessed_inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=2,
        padding='same', activation='sigmoid'
    )  # 384x384 -> 768x768

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def upscale_layer(filters: int):
    """Layer was copied from the tensorflow_examples."""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters, 3, strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
    )
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())

    return result