import nls_utils as utils
import numpy as np
import pandas as pd

a = np.array([[-1.0, 0.2],
              [-2.0, -1.2],
              [3.0, 2.0]])
d = pd.DataFrame({"x": [-1.0, -2.0, 3.0],
                  "y": [0.2, -1.2, 2.0]},
                 index=["a", "b", "c"])
idx2label = {0: "x",
             1: "y"}

# array input: ['y' 'y' 'x']
ay = utils.logits2labels(a, idx2label)
print(ay)

# dataframe input:
#  labels
#a      y
#b      y
#c      x
dy = utils.logits2labels(d)
print(dy)
