import argparse
import audeer
import audiofile
from glob import glob
import numpy as np
import os
import pandas as pd
import pickle
import sys
from tqdm import tqdm

from speech_detector import SpeechDetector
import utils
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(cwd, "..")))
from nonlex_speech_clf import (
    NlsClf,
    FormatConverter,
    TextGridProc
)


''' script to segment and label CGs and FPs within pause segments
of the Budapest Games Corpus (work in progress!). The pause segments
have been derived before by means of ASR that was treating nonlexical
speech as pauses.

Additional dependencies:
scipy

Call:
$ python process_bgc.py -pa DIR_AUDIO -pt DIR_TEXTGRIDS \
         -po DIR_OUTPUT -pc DIR_CACHE

Outputs TextGrids with additional tiers NLS1 and NLS2 (one per channel),
each containing additional segments with labels "cg" and "fp"

Comments:
To identify CG and FP segments the pause detector of CoPaSul was inverted
to a chunk detector
'''

def process_bgc(
        path_audio: str,
        path_tg: str,
        path_cache: str,
        path_output: str
) -> None:

    _ = audeer.mkdir(path_cache)
    _ = audeer.mkdir(path_output)
   
    ff_tg = glob(f"{path_tg}/**/*.TextGrid", recursive=True)
    ff_wav = glob(f"{path_audio}/**/*.wav", recursive=True)
    
    # input tiers per channel
    tiers_in = ["SPK1", "SPK2"]
    # output tiers
    tiers = ["NLS1", "NLS2"]

    # (1) find speech chunks within pause segments
    data = bgc_chunks(ff_tg, ff_wav, tiers_in, path_cache)
    
    # (2) apply model
    print("NLS prediction ...")
    nls = NlsClf()
    ans = {}
    for c in [0, 1]:
        print(f"processing channel {c} ...")
        ans[c] = nls.process_index(
            index=data[c].index,
            channel=c,
            num_jobs=5,
            return_type="labels",
            cache_path=os.path.join(path_cache, f"nls-{c}.pkl")
        )
        
    # (3) add labels to TextGrids, output to output_dir
    tgp = TextGridProc()
    fconv = FormatConverter(source_format="audformat",
                            target_format="textgrid")
    for f_tg, f_wav in zip(ff_tg, ff_wav):
        # read tg
        tg = tgp.read(f_tg)
        
        # replace dummy content of tiers nls1 and nls2
        # with model predictions
        for c, t in enumerate(tiers):
            ff = ans[c].index.get_level_values("file")
            ii = np.where(ff == f_wav)[0]
            x = ans[c].iloc[ii]
            tg = fconv.convert(x=x, tg=tg, tiername=t)
            
        # write tg
        fo = os.path.join(path_output, f"{utils.stm(f_tg)}.TextGrid")
        tgp.write(tg, fo)

        
def bgc_chunks(
        ff_tg: list, ff_wav: list, tiers_in: list, path_cache: str
) -> dict:

    r"""finds speech chunks in pause intervals

    Args:
        ff_tg: TextGrid file names
        ff_wav: audio file names
        tiers_in: input tier names
        path_cache: cache directory

    Returns:
        dict with channel keys and audormat tables
    
    """
    
    # TextGrid processor
    tgp = TextGridProc()

    fo = os.path.join(path_cache, "chunks.pkl")
    if os.path.isfile(fo):
        print(f"reading chached chunks from {fo} ...")
        with open(fo, "rb") as h:
            return pickle.load(h)
    
    #     concat to 2 audformat dataframes, one per channel
    spd_param = {
        "e_rel": 0.5,  # .2
        "l": 0.15,
        "l_ref": 5,    # 5
        "n": -1,
        "fbnd": 0,
        "min_pau_l": 0.2,
        "min_chunk_l": 0.1,
        "flt_btype": "low",
        "flt_f": 8000,
        "flt_ord": 5
    }
    data = {0: {"file": [], "start": [], "end": [], "labels": []},
            1: {"file": [], "start": [], "end": [], "labels": []}}
    spd = SpeechDetector(**spd_param)
    for j in tqdm(range(len(ff_tg)), desc="chunking"):
        f_tg, f_wav = ff_tg[j], ff_wav[j]
        sig_stereo, sr = audiofile.read(f_wav)
        tg = tgp.read(f_tg)
        for c, tn in enumerate(tiers_in):
            sig = sig_stereo[c]
            tier = tgp.tier(tg, tn)
            t, lab = tgp.tier2table(tier, {"skip_empty": False})
            t_chunks = np.array([])
            for i in range(len(lab)):
                if len(lab[i]) > 0:
                    # no pause
                    continue
                tt = utils.time2idx(t[i, :], sr)
                y = sig[tt[0]:tt[1]]
                tc = spd.process_signal(y, sr, onset=tt[0])
                n = tc.shape[0]
                if n == 0:
                    continue
                data[c]["file"].extend([f_wav] * n)
                data[c]["start"].extend(tc[:, 0].tolist())
                data[c]["end"].extend(tc[:, 1].tolist())
                data[c]["labels"].extend(["x"] * n)

    # convert to audformat
    fconv = FormatConverter(source_format="table",
                            target_format="audformat")
    for c in data:
        data[c] = fconv.convert(x=pd.DataFrame(data[c]))
                
    with open(fo, "wb") as h:
        pickle.dump(data, h)

    return data
                

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("label Budapest Games Corpus with cg and fp")

    parser.add_argument('-pa', '--path_audio',
                        help='directory with audio files',
                        type=str, required=True)
    parser.add_argument('-pt', '--path_tg',
                        help='directory with TextGrid files',
                        type=str, required=True)
    parser.add_argument('-pc', '--path_cache',
                        help='cache directory',
                        type=str, required=True)
    parser.add_argument('-po', '--path_output',
                        help='output directory for extended TextGrids',
                        type=str, required=True)
    
    kwargs = vars(parser.parse_args())
    process_bgc(**kwargs)
