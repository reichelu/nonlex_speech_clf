import copy
import math
import numpy as np
import re
import scipy.signal as sis
import sys
import utils

class SpeechDetector():

    r"""detect speech chunks in pauses"""
    
    def __init__(self, e_rel=0.2, l=0.15, l_ref=5, n=-1,
                 fbnd=0, min_pau_l=0.2, min_chunk_l=0.1,
                 flt_btype="low", flt_f=8000, flt_ord=5):

        super().__init__()
        self.param = {
            "e_rel": e_rel,
            "l": l,
            "l_ref": l_ref,
            "n": n,
            "fbnd": fbnd,
            "min_pau_l": min_pau_l,
            "min_chunk_l": min_chunk_l,
            "flt": {
                "btype": flt_btype,
                "f": flt_f,
                "ord": flt_ord
            }
        }

        
    def process_signal(self, y, sr, onset=0.0):

        '''
        Args:
        y: (np.array) signal
        sr: (int) sampling rate
        onset: (float) time onset to be added to chunk time stamps
        Returns:
        tc: (np.array) [[start, end], ...] in sec
        '''
        
        y -= np.mean(y)
        
        opt = copy.copy(self.param)
        opt["flt"]["fs"] = sr
        opt["ons"] = onset
        opt["fs"] = sr

        pau = pau_detector(y, opt)
        return pau["tc"]


def pau_detector(s, opt={}):

    '''
    pause detection
    
    Args:
      s - mono signal
      opt['fs']  - sample frequency
         ['ons'] - idx onset <0> (to be added to time output)
         ['flt']['f']     - filter options, boundary frequencies in Hz
                            (2 values for btype 'band', else 1): <8000> (evtl. lowered by fu_filt())
                ['btype'] - <'band'>|'high'|<'low'>
                ['ord']   - butterworth order <5>
                ['fs']    - (internally copied)
         ['l']     - analysis window length (in sec)
         ['l_ref'] - reference window length (in sec)
         ['e_rel'] - min energy quotient analysisWindow/referenceWindow
         ['fbnd']  - True|<False> assume pause at beginning and end of file
         ['n']     - <-1> extract exactly n pauses (if > -1)
         ['min_pau_l'] - min pause length <0.5> sec
         ['min_chunk_l'] - min inter-pausal chunk length <0.2> sec
         ['force_chunk'] - <False>, if True, pause-only is replaced by chunk-only
         ['margin'] - <0> time to reduce pause on both sides (sec; if chunks need init and final silence)
    
    Returns:
       pau['tp'] 2-dim array of pause [on off] (in sec)
          ['tpi'] 2-dim array of pause [on off] (indices in s = sampleIdx-1 !!)
          ['tc'] 2-dim array of speech chunks [on off] (i.e. non-pause, in sec)
          ['tci'] 2-dim array of speech chunks [on off] (indices)
          ['e_ratio'] - energy ratios corresponding to pauses in ['tp'] (analysisWindow/referenceWindow)
    '''

    if 'fs' not in opt:
        sys.exit('pau_detector: opt does not contain key fs.')
    dflt = {'e_rel': 0.0767, 'l': 0.1524, 'l_ref': 5, 'n': -1, 'fbnd': False, 'ons': 0, 'force_chunk': False,
            'min_pau_l': 0.4, 'min_chunk_l': 0.2, 'margin': 0,
            'flt': {'btype': 'low', 'f': np.asarray([8000]), 'ord': 5}}
    opt = utils.opt_default(opt, dflt)
    opt['flt']['fs'] = opt['fs']

    # removing DC, low-pass filtering
    flt = fu_filt(s, opt['flt'])
    y = flt['y']

    # pause detection for >=n pauses
    t, e_ratio = pau_detector_sub(y, opt)

    if len(t) > 0:

        # extending 1st and last pause to file boundaries
        if opt['fbnd'] == True:
            t[0, 0] = 0
            t[-1, -1] = len(y) - 1

        # merging pauses across too short chunks
        # merging chunks across too small pauses
        if (opt['min_pau_l'] > 0 or opt['min_chunk_l'] > 0):
            t, e_ratio = pau_detector_merge(t, e_ratio, opt)

        # too many pauses?
        # -> subsequently remove the ones with highest e-ratio
        if (opt['n'] > 0 and len(t) > opt['n']):
            t, e_ratio = pau_detector_red(t, e_ratio, opt)

    # speech chunks
    tc = pau2chunk(t, len(y))

    # pause-only -> chunk-only
    if (opt['force_chunk'] == True and len(tc) == 0):
        tc = copy.deepcopy(t)
        t = np.asarray([])
        e_ratio = np.asarray([])

    # add onset
    t = t + opt['ons']
    tc = tc + opt['ons']

    # return dict
    # incl fields with indices to seconds (index+1=sampleIndex)
    pau = {'tpi': t, 'tci': tc, 'e_ratio': e_ratio}
    pau['tp'] = utils.idx2sec(t, opt['fs'])
    pau['tc'] = utils.idx2sec(tc, opt['fs'])

    return pau


def pau_detector_sub(y, opt):

    '''
    called by pau_detector
    
    Args:
       as for pau_detector
    
    Returns:
       t: (np.array) [[on off], ...]
       e_ratio: (np.array)
    '''

    # settings
    # reference window span
    rl = math.floor(opt['l_ref'] * opt['fs'])
    
    # signal length
    ls = len(y)

    # min pause length
    ml = opt['l']*opt['fs']

    # global rmse and pause threshold
    e_rel = copy.deepcopy(opt['e_rel'])

    # global rmse
    # as fallback in case reference window is likely to be pause
    # almost-zeros excluded (cf percentile) since otherwise pauses
    # show a too high influence, i.e. lower the reference too much
    # so that too few pauses detected
    ya = np.abs(y)
    qq = np.percentile(ya, [50])
    e_glob = utils.rmsd(ya[ya > qq[0]])
    t_glob = opt['e_rel'] * e_glob

    # stepsize
    sts = max([1, math.floor(0.05*opt['fs'])])
    
    # energy calculation in analysis and reference windows
    wopt_en = {'win': ml, 'rng': [0, ls]}
    wopt_ref = {'win': rl, 'rng': [0, ls]}

    # loop until opt.n criterion is fulfilled
    # increasing energy threshold up to 1
    while e_rel < 1:
        # pause [on off], pause index
        t = []
        j = 0
        # [e_y/e_rw] indices as in t
        e_ratio = []
        i_steps = np.arange(1, ls, sts)
        for i in i_steps:

            # window
            yi = utils.windowing_idx(i, wopt_en)
            e_y = utils.rmsd(y[yi])

            # energy in reference window
            e_r = utils.rmsd(y[utils.windowing_idx(i, wopt_ref)])

            # take overall energy as reference if reference window is pause
            if (e_r <= t_glob):
                e_r = e_glob

            # if rmse in window below threshold
            if e_y <= e_r * e_rel:
                yis = yi[0]
                yie = yi[-1]
                if len(t) - 1 == j:
                    # values belong to already detected pause
                    if len(t) > 0 and yis < t[j][1]:
                        t[j][1] = yie
                        # evtl. needed to throw away superfluous
                        # pauses with high e_ratio
                        e_ratio[j] = np.mean([e_ratio[j], e_y / e_r])
                    else:
                        t.append([yis, yie])
                        e_ratio.append(e_y / e_r)
                        j = j+1
                else:
                    t.append([yis, yie])
                    e_ratio.append(e_y / e_r)
                    
        # (more than) enough pauses detected?
        if len(t) >= opt['n']:
            break
        e_rel = e_rel + 0.1
        
    if opt['margin'] == 0 or len(t) == 0:
        return np.array(t), np.array(e_ratio)

    # shorten pauses by margins
    mar = int(opt['margin'] * opt['fs'])
    tm, erm = [], []

    for i in utils.idx_a(len(t)):
        
        # only slim non-init and -fin pauses
        if i > 0:
            ts = t[i][0] + mar
        else:
            ts = t[i][0]
        if i < len(t) - 1:
            te = t[i][1] - mar
        else:
            te = t[i][1]

        # pause disappeared
        if te <= ts:
            
            # ... but needs to be kept
            if opt['n'] > 0:
                tm.append(t[i])
                erm.append(e_ratio[i])
            continue
        
        # pause still there
        tm.append([ts, te])
        erm.append(e_ratio[i])

    return np.array(tm), np.array(erm)


def pau_detector_red(t, e_ratio, opt):

    ''' remove pauses with highes e_ratio
    (if number of pause restriction applies)

    Args:
    t: (np.array) [[on, off], ...]
    e_ratio: (np.array) of energy ratios
    opt: (dict)

    Returns:
    t: (np.array)
    e_ratio: (np.array)

    '''
    
    # keep boundary pauses
    if opt['fbnd'] == True:
        n = opt['n'] - 2
        bp = np.concatenate((np.array([t[0,]]), np.array([t[-1,]])), axis=0)
        ii = np.arange(1, len(t)-1, 1)
        t = t[ii,]
        e_ratio = e_ratio[ii]
    else:
        n = opt['n']
        bp = np.array([])

    if n == 0:
        t = []

    # remove pause with highest e_ratio
    while len(t) > n:
        i = np.argmax(e_ratio)
        aran = np.arange(1, len(e_ratio), 1)
        j = np.where(aran != i)[0]
        
        t = t[j,]
        e_ratio = e_ratio[j]

    # re-add boundary pauses if removed
    if opt['fbnd'] == True:
        if len(t) == 0:
            t = np.concatenate(
                (np.array([bp[0,]]), np.array([bp[1,]])), axis=0)
        else:
            t = np.concatenate(
                (np.array([bp[0,]]), np.array([t]), np.array([bp[1,]])), axis=0)

    return t, e_ratio


def pau_detector_merge(t, e, opt):

    '''
    merging pauses across too short chunks
    merging chunks across too small pauses
    
    Args:
      t: (np.array) [[on off]...] of pauses
      e: (np.array) [e_rat ...]
    
    Returns:
      t [[on off]...] merged
      e [e_rat ...] merged (simply mean of merged segments taken)
    '''

    # min pause and chunk length in samples
    mpl = utils.sec2smp(opt['min_pau_l'], opt['fs'])
    mcl = utils.sec2smp(opt['min_chunk_l'], opt['fs'])

    # merging chunks across short pauses
    tm = []
    em = []
    for i in utils.idx_a(len(t)):
        if ((t[i, 1]-t[i, 0] >= mpl) or
                (opt['fbnd'] == True and (i == 0 or i == len(t)-1))):
            tm.append(list(t[i, :]))
            em.append(e[i])

    # nothing done in previous step?
    if len(tm) == 0:
        tm = copy.deepcopy(t)
        em = copy.deepcopy(e)
    if len(tm) == 0:
        return t, e

    tm = np.array(tm)
    em = np.array(em)
    
    # merging pauses across short chunks
    tn = list([tm[0, :]])
    en = [em[0]]
    if (tn[0][0] < mcl):
        tn[0][0] = 0.0
    for i in np.arange(1, len(tm), 1):
        if (tm[i, 0] - tn[-1][1] < mcl):
            tn[-1][1] = tm[i, 1]
            en[-1] = np.mean([en[-1], em[i]])
        else:
            tn.append(list(tm[i, :]))
            en.append(em[i])

    return np.array(tn), np.array(en)


def pau2chunk(t, l):

    '''
    pause to chunk intervals
    
    Args:
       t [[on off]] of pause segments (indices in signal)
       l length of signal vector
    
    Returns:
       tc [[on off]] of speech chunks
    '''

    if len(t) == 0:
        return np.array([[0, l - 1]])
    if t[0, 0] > 0:
        tc = np.array([[0, t[0, 0] - 1]])
    else:
        tc = np.array([])
    for i in np.arange(0, len(t) - 1, 1):
        if t[i, 1] < t[i + 1, 0] - 1:
            tc = utils.push(tc, [t[i, 1] + 1, t[i + 1, 0] - 1])
    if t[-1, 1] < l - 1:
        tc = utils.push(tc, [t[-1, 1] + 1, l - 1])
        
    return tc


def fu_filt(y, opt):

    '''
    wrapper around Butter filter
    
    Args:
      y: (np.array) 1-dim vector
      opt: (dict)
        opt['fs'] - sample rate
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

    # check f < fs / 2
    if (opt['btype'] == 'low' and opt['f'] >= opt['fs'] / 2):
        opt['f'] = opt['fs'] / 2 - 100
    elif (opt['btype'] == 'band' and opt['f'][1] >= opt['fs'] / 2):
        opt['f'][1] = opt['fs'] / 2 - 100
    fn = opt['f'] / (opt['fs'] / 2)
    b, a = sis.butter(opt['ord'], fn, btype=opt['btype'])

    yf = sis.filtfilt(b, a, y)
    return {'y': yf, 'b': b, 'a': a}
