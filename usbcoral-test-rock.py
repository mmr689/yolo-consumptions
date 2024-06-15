from tflite_runtime.interpreter import load_delegate

try:
    delegate = load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')
    print("Delegate loaded successfully!")
except Exception as e:
    print(f"Failed to load delegate: {e}")
