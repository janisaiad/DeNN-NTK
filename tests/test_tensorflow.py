import pytest


def test_import_tensorflow():
    """test if tensorflow can be imported"""
    import tensorflow as tf
    assert tf is not None, "TensorFlow not found"

def test_tensorflow_gpu():
    """test if tensorflow can detect and use GPU"""
    import tensorflow as tf
    devices = tf.config.list_physical_devices()
    print(devices)
    gpu_devices = [d for d in devices if d.device_type == 'GPU']
    assert len(gpu_devices) > 0, "No GPU devices found"
    """
    [GpuDevice(id=0, platform='cuda', process_index=0, visible_device_list=['0'], memory_limit=None), GpuDevice(id=1, platform='cuda', process_index=0, visible_device_list=['1'], memory_limit=None)]
    """

if __name__ == "__main__":
    test_import_tensorflow()
    test_tensorflow_gpu()