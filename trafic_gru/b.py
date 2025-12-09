import torch
if torch.cuda.is_available():
    print("Using GPU for computation")
else:
    print("Using CPU for computation")