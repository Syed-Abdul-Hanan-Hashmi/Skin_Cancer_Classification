import tensorflow as tf
import numpy as np

# Exact layer mapping for your architectures
LAYER_CONFIG = {
    "DenseNet121": "conv5_block16_2_conv",
    "ResNet50V2": "post_relu",
    "MobileNetV2": "out_relu",
    "Custom CNN": "conv2d_2",
    "ResNet50": "conv5_block3_out"
}

def get_gradcam_heatmap(img_array, model, model_type):
    target_layer_name = LAYER_CONFIG.get(model_type)
    
    # Identify the base model layer within the functional wrapper
    try:
        base_engine = next(l for l in model.layers if any(x in l.name.lower() for x in ['densenet', 'resnet', 'mobilenet', 'sequential']))
    except StopIteration:
        base_engine = model

    grad_model = tf.keras.models.Model(
        [base_engine.inputs], 
        [base_engine.get_layer(target_layer_name).output, base_engine.output]
    )

    with tf.GradientTape() as tape:
        # Pass input through any initial rescaling/augmentation layers
        x = img_array
        for layer in model.layers:
            if layer == base_engine: break
            x = layer(x)
            
        conv_outputs, engine_preds = grad_model(x)
        class_idx = tf.argmax(engine_preds[0])
        loss = engine_preds[:, class_idx]

    # Calculate gradients and create the heatmap
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()