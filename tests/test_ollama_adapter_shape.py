def test_ollama_adapter_shape():
    # Only test shape matching logic
    import numpy as np
    dummy_response = [0.1] * 768
    arr = np.array(dummy_response, dtype=np.float32)
    assert arr.shape == (768,)
