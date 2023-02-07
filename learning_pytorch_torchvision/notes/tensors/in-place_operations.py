import torch
import numpy as np

tensor = torch.ones(4,4)

#in-place operations are operations that store the result back into the operand itself. they are denoted by a _ suffix. for example: {x.copy_(y)} or {x.t_()} will change x
print(tensor, "\n")
tensor.add_(5)
print(tensor)

#NOTE: in place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.