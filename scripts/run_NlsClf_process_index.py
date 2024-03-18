import numpy as np
import os
import pandas as pd
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(cwd, "..")))

from nonlex_speech_clf import NlsClf
from nonlex_speech_clf import FormatConverter

# read table, add absolute path
cwd = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(cwd, "..", "assets")
data = pd.read_csv(os.path.join(assets_path, "index.csv"), header=0)
def add_assets_path(x):
    return os.path.join(assets_path, x)
data["file"] = data["file"].apply(add_assets_path)

# convert to audformat
fc = FormatConverter(
    source_format="table",
    target_format="audformat"
)
data = fc.convert(data)

# run model
nls = NlsClf()
y = nls.process_index(data.index)
print(y)

