import torch, os
f = '/root/.cache/huggingface/hub/models--BioMistral--BioMistral-7B/snapshots/9a11e1ffa817c211cbb52ee1fb312dc6b61b40a5/pytorch_model.bin'
print(f'Size: {os.path.getsize(f)/1e9:.2f} GB')
try:
    torch.load(f, map_location='cpu', weights_only=False)
    print('OK')
except Exception as e:
    print(f'FAILED: {e}')
