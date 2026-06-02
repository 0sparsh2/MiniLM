import torch
import numpy as np
import struct

def test_export():
    tensor = torch.arange(-16, 16, dtype=torch.float32).unsqueeze(0) # shape [1, 32]
    tensor[0, 0] = -32.0 # inject an outlier
    
    blocks = tensor.reshape(-1, 32)
    abs_max = blocks.abs().max(dim=1).values
    d = abs_max / 7.0
    
    q = torch.round(blocks / d.unsqueeze(1)).clamp(-8, 7).to(torch.int8)
    q_shifted = (q + 8).to(torch.uint8)
    
    q_even = q_shifted[:, 0::2]
    q_odd = q_shifted[:, 1::2]
    packed = ((q_even << 4) | q_odd).to(torch.uint8)
    
    dt = np.dtype([('d', np.float32), ('qs', np.uint8, (16,))])
    arr = np.empty(blocks.shape[0], dtype=dt)
    arr['d'] = d.numpy()
    arr['qs'] = packed.numpy()
    
    with open("test.bin", "wb") as f:
        f.write(arr.tobytes())

test_export()
