import os
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_SAVE_IR"] = "1"
os.environ["TORCHINDUCTOR_DUMP_CODE"] = "1"

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = False

model = torch.nn.Linear(15,15)
model.eval()

compiled = torch.compile(model, backend="inductor", fullgraph=True)
compiled(torch.randn(1, 15))
