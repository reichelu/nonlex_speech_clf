import copy
import math
import numpy as np
import re
import scipy.signal as sis
import sys
from typing import Union
import utils

class SpeechDetector():

    def __init__(
            self,
            local_ratio: float = 2.0,
            global_ratio: float = 0.1,
            l: float = 0.1,
            l_ref: float = 0.5,
            min_l: float = 0.1,
            flt_btype: str = "low",
            flt_f: Union[int, list, np.array] = 8000,
            flt_ord: int = 5):

        r"""detect short speech chunks in pauses
        
        Args:
            l: analysis window length
            l_ref: reference window length
            min_l: min chunk duration in sec
            flt_btype: Butterworth filter bandpass type
            flt_f: filter cutoff frequency/ies
            flt_ord: filter order

        Comments:
            speech chunk criteria
                - higher energy than reference energy times local ratio
                - higher energy than overall reference energy times global ratio
                - min length

        """
        
        super().__init__()
        
        self.param = {
            "local_ratio": local_ratio,
            "global_ratio": global_ratio,
            "l": l,
            "l_ref": l_ref,
            "min_l": min_l,
            "flt": {
                "btype": flt_btype,
                "f": flt_f,
                "ord": flt_ord
            }
        }

        
    def get_global_reference(self, y: np.array, threshold_prct: int = 50) -> float:

        r"""
        get energy reference value on longer signal, for cases where later only short
        signals are to be processed

        Args:
            y: signal
            threshold_prct: percentile for abs amplitude values above which to calculate
                the reference for the global energy ratio

        Returns:
            lower energy threshold
        """
        
        ya = np.abs(y)
        qq = np.percentile(ya, [50])

        return utils.rmsd(ya[ya > qq[0]])

    
    def process_signal(self, y, sr, global_reference=0.0, onset=0.0):

        r"""
        Args:
            y: (np.array) signal
            sr: (int) sampling rate
            global_reference: (float) energy reference times global_ratio
                to be superseeded
            onset: (float) onset in seconds to be added to chunk time stamps
        
        Returns:
            t: (np.array) [[start, end], ...] in sec
        """
        
        opt = self.param
        opt["flt"]["sr"] = sr
        
        y -= np.mean(y)

        # low-pass filtering
        ret = fu_filt(y, opt['flt'])
        y = ret['y']

        # window
        l = math.floor(opt['l'] * sr)
        # reference window
        rl = math.floor(opt['l_ref'] * sr)

        # signal length
        ls = len(y)

        # min chunk length
        ml = opt['l'] * sr

        # stepsize
        sts = max([1, math.floor(0.05 * sr)])

        # energy calculation in analysis and reference windows
        ana_wopt = {'win': l, 'rng': [0, ls]}
        ref_wopt = {'win': rl, 'rng': [0, ls]}
        
        # chunk [on off], pause index
        t = []
        
        for i in np.arange(1, ls, sts):

            # energy analyis window
            yi = utils.windowing_idx(i, ana_wopt)
            yen = utils.rmsd(y[yi])

            # local reference energy
            ri = utils.windowing_idx(i, ref_wopt)
            ren = utils.rmsd(y[ri])
            
            # criteria met
            if (yen >= ren * opt["local_ratio"] and
                yen >= global_reference * opt["global_ratio"]):
                if len(t) > 0 and yi[0] < t[-1][1]:
                    # overlap with already detected pause
                    t[-1][1] = yi[-1]
                else:
                    t.append([yi[0], yi[-1]])

        if len(t) == 0:
            return np.array([])
                    
        # remove too short chunks
        t = np.array(t)
        d = np.diff(t, axis=1)
        i = np.where(d >= ml)[0]
        t = t[i, :]

        # conversion to seconds, add onset
        t = utils.idx2sec(t, sr)
        t = t + onset
        
        return t


def fu_filt(y, opt):

    '''
    wrapper around Butter filter
    
    Args:
      y: (np.array) 1-dim vector
      opt: (dict)
        opt['sr'] - sample rate
           ['f']  - scalar (high/low) or 2-element vector (band) of boundary freqs
           ['order'] - order
           ['btype'] - band|low|high; all other values: signal returned as is
    
    Returns:
      flt: (dict)
        flt['y'] - filtered signal
           ['b'] - coefs
           ['a']
    '''

    # do nothing
    if not re.search(r'^(high|low|band)$', opt['btype']):
        return {'y': y, 'b': np.array([]), 'a': np.array([])}

    # check f < sr / 2
    if (opt['btype'] == 'low' and opt['f'] >= opt['sr'] / 2):
        opt['f'] = opt['sr'] / 2 - 100
    elif (opt['btype'] == 'band' and opt['f'][1] >= opt['sr'] / 2):
        opt['f'][1] = opt['sr'] / 2 - 100
    fn = opt['f'] / (opt['sr'] / 2)
    b, a = sis.butter(opt['ord'], fn, btype=opt['btype'])

    yf = sis.filtfilt(b, a, y)
    return {'y': yf, 'b': b, 'a': a}
