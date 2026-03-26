import os
import tensorflow as tf
import gc

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_model_safely(model_path):
    """
    Clears memory and loads model specifically for CPU execution.
    """
    if not os.path.exists(model_path):
        print(f"❌ ERROR: File not found at {model_path}")
        return None

    tf.keras.backend.clear_session()
    gc.collect()
    
    print(f"🖥️ Loading model for CPU: {model_path}")
    # compile=False makes loading much faster for inference
    model = tf.keras.models.load_model(model_path, compile=False)
    return model