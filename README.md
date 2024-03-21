# Non-lexical speech classifier

This project contains code to distinguish communicative grunts from filled pauses.

## Introduction

Conversation grunts, short non-lexical utterances can be further subdivided into
* communicative grunts (`cg`), i.e. short non-lexical utterances with communicative intent, and
* filled pauses (`fp`), which are rather by-products of speech planning.

This project contains code to distinguish these two classes `cg` and `fp` by applying a wav2vec2 transformer model exported to ONNX.

See [this reference](http://real.mtak.hu/159991/1/beszkut_speechresearch_2023_proceedings.pdf#page=91) for more details on model architecture and evaluation.

The model has been trained on mono signals with a sampling rate of 16 kHz. The interface in this project takes care that the input signal is converted accordingly.

## Limitations

* the code does not provide speech segmentation, but only classification of pre-segmented speech intervals
* the currently underlying model was trained on Hungarian data only

## Installation

* install requirements

```bash
$ virtualenv --python="/usr/bin/python3" nls
$ source nls/bin/activate
(nls) $ pip install -r requirements.txt
```

* download model from Zenodo
    * record [URL](https://zenodo.org/records/10833104)
    * model DOI: `10.5281/zenodo.10833104`
* and unzip in `model/` folder

```bash
$ cd model/
$ wget https://zenodo.org/records/10833104/files/nonlex_speech_model.zip?download=1 -O nonlex_speech_model.zip
$ unzip nonlex_speech_model.zip
$ rm nonlex_speech_model.zip
```

## Usage

### Process a signal

* see [example script](https://github.com/reichelu/nonlex_speech_clf/blob/main/scripts/run_NlsClf_process_signal.py)

```python

import numpy as np
from nonlex_speech_clf import NlsClf

# create dummy signal with duration 1s
sampling_rate = 16000
signal = np.random.rand(sampling_rate)

nls = NlsClf()
y = nls.process_signal(signal, sampling_rate)
print(y)
```

### Process a DataFrame index

* see [example script](https://github.com/reichelu/nonlex_speech_clf/blob/main/scripts/run_NlsClf_process_index.py)


## Reference

If you use this model for your studies please cite:

Reichel, U.D., Kohári, A., Mády, K. (2023). Acoustics and prediction of non-lexical speech in the Budapest Games Corpus. In: Proc. Speech Research Conference, [pdf](http://real.mtak.hu/159991/1/beszkut_speechresearch_2023_proceedings.pdf#page=91)

```
@InProceedings{rkm2023,
  author =       {Reichel, U.D. and Koh\'ari, A. and M\'ady, K.},
  title =        {Acoustics and prediction of non-lexical speech in the {B}udapest {G}ames {C}orpus},
  booktitle = {Proc. Speech Research Conference},
  year =         {2023},
  address =      {Budapest, Hungary}
}
```

## Funding

* Funder: National Research, Development and Innovation Office
* Project ID: NKFIH K 135038
* Project title: Prosodic structure and sentence types by using large speech databases supported by deep learning techniques
* [Project URL](https://nytud.hu/en/tender/prozodiai-szerkezet-es-mondattipusok-vizsgalata-nagy-beszedadatbazisokon-mely-tanulasi-tamogatassal)
