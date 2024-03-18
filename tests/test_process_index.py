import numpy as np
import os
import pandas as pd
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(cwd, "..")))

from nonlex_speech_clf import NlsClf
from nonlex_speech_clf import FormatConverter

def test_process_index():

    cwd = os.path.dirname(os.path.abspath(__file__))
    assets_path = os.path.join(cwd, "..", "assets")
    data = pd.read_csv(os.path.join(assets_path, "index.csv"), header=0)
    def add_assets_path(x):
        return os.path.join(assets_path, x)
    data["file"] = data["file"].apply(add_assets_path)

    fc = FormatConverter(
        source_format="table",
        target_format="audformat"
    )
    data = fc.convert(data)

    nls = NlsClf()
    y = nls.process_index(data.index)

    ref = np.array([[-2.726331, 3.134038],
                    [1.586750, -1.858605]])

    assert np.allclose(y, ref), "test process index failed"


if __name__ == "__main__": 
    test_process_index()
