import audformat
import copy
import numpy as np
import pandas as pd
import re
from typing import Any, Tuple, Union

class FormatConverter(object):

    def __init__(
            self,
            source_format: str = "table",
            target_format: str = "audformat"
    ):

        r"""Format conversions

        Args:
           source_format: "table", "audformat", "textgrid"
           target_format: "table", "audformat", "textgrid"
        
        Returns:
           converted output

        Formats:
           "table": pd.DataFrame with columns "file" (file-level processing)
                or "file", "start", "end" (segment-level processing).
                Time one- and offsets in "start" and "end" provided in seconds
                For conversions from TextGrid:
                    - an addtional column "labels" is outputted
                    - for point tiers the "end" column is dropped
                For conversions from audformat:
                    - all column names are kept
           "audformat": pd.Index "file" (file-level processing)
                or pd.MultiIndex "file", "start", "end" (segment-level processing).
                Time values are in timedelta format.
                pd.DataFrame with one of the two indices described above.
                For conversions from TextGrid:
                    - an addtional column "labels" is outputted
                    - for point tiers the "start" and "end" column will be
                        assigned the same values
                For conversions from audformat:
                    - all column names but file, start, end are kept
           "textgrid": textgrid dict
                For conversions from table or audformat:
                    - the labels are read from the column columnname
        """

        super().__init__()
        
        self.source_format = source_format
        self.target_format = target_format
        
    def convert(
            self,
            x: Union[dict, pd.DataFrame, pd.Index, pd.MultiIndex],
            tg: dict = None,
            tiername: str = None,
            columnname: str = "labels",
            filename: str = None,
            source_format: str = None,
            target_format: str = None
    ) -> Union[dict, pd.DataFrame]:

        """ conversions
        
        Args:
            x: input dict, dataframe, or index
            tg: textgrid dict (in case table should be added as new tier
                to exisiting TextGrid
            tiername: tier name (required for all conversions from or to TextGrids)
            columnname: name of column (required for all conversions to TextGrids)
            filename: file name (if to be outputted in table; required for
                source_format=="textgrid" and target_format=="audformat"
            source_format: if self.source_format should be overwritten
            target_format: if self.target_format should be overwritten
        Returns:
            dataframe or Textgrid dict

        """

        if source_format is None:
            source_format = self.source_format
        if target_format is None:
            target_format = self.target_format
        
        if "textgrid" in [source_format, target_format]:
            assert tiername is not None, "tiername has to be specified"

            if (source_format == "textgrid" and
                target_format == "audformat"):
                assert filename is not None, "filename has to be specified"

            elif target_format == "textgrid":
                assert columnname is not None, "column name has to be specified"
                
        if source_format == "audformat":
            if target_format == "table":
                return self.audformat_to_table(x)
            elif target_format == "textgrid":
                return self.audformat_to_textgrid(
                    x=x,
                    tiername=tiername,
                    columnname=columnname,
                    tg=tg
                )
        elif source_format == "table":
            if target_format == "audformat":
                return self.table_to_audformat(x)
            elif target_format == "textgrid":
                return self.table_to_textgrid(
                    x=x,
                    tiername=tiername,
                    columnname=columnname,
                    tg=tg
                )
        elif source_format == "textgrid":
            if target_format == "audformat":
                return self.textgrid_to_audformat(
                    x=x,
                    tiername=tiername,
                    filename=filename,
                    columnname=columnname
                )
            elif target_format == "table":
                return self.textgrid_to_table(
                    x,
                    tiername=tiername,
                    columnname=columnname,
                    filename=filename
                )
            
        return x

    def table_to_audformat(
            self,
            x: pd.DataFrame
    ) -> pd.DataFrame:

        r""" converts dataframe to audformat; returns pd.DataFrame """

        x = copy.deepcopy(x)
        
        cols = x.columns
        assert "file" in cols, "table does not contain file column"
        
        if "start" in cols and "end" in cols:
            index = audformat.segmented_index(
                files=x["file"].to_numpy(),
                starts=x["start"].to_numpy(),
                ends=x["end"].to_numpy()
            )
        else:
            index = audformat.filewise_index(
                files=x["file"].to_numpy()
            )

        if audformat.is_segmented_index(index):
            x.drop(columns=["file", "start", "end"], inplace=True)
        else:
            x.drop(columns=["file"], inplace=True)

        x.index = index

        return x
    
    def audformat_to_table(
            self,
            x: Union[pd.DataFrame, pd.Index, pd.MultiIndex]
    ) -> pd.DataFrame:

        r""" converts audformat to table; returns DataFrame """

        x = copy.deepcopy(x)
        
        if type(x) is pd.DataFrame:
            index = x.index
        else:
            index = x

        files = index.get_level_values("file")
        starts, ends = None, None
        if audformat.is_segmented_index(index):    
            starts = index.get_level_values("start").total_seconds().to_numpy()
            ends = index.get_level_values("end").total_seconds().to_numpy()
        x.reset_index(inplace=True)
        x["file"] = files
        if starts is not None:
            x["start"] = starts
            x["end"] = ends
            
        return x

    def textgrid_to_table(
            self,
            x: dict,
            tiername: str,
            columnname: str = "labels",
            filename: str = None
    ) -> pd.DataFrame:
        
        tgp = TextGridProc()
        tier = tgp.tier(x, tiername)
        x, lab = tgp.tier2table(tier)

        if x.ndim == 2:
            tab = {"start": x[:, 0],
                   "end": x[:, 1],
                   columnname: lab}
        else:
            tab = {"start": x,
                   columnname: lab}
        if filename:
            tab["file"] = filename
            
        return pd.DataFrame(tab)    
        
    def textgrid_to_audformat(
            self,
            x: dict,
            tiername: str,
            filename: str,
            columnname: str = "labels"
    ) -> pd.DataFrame:

        return self.table_to_audformat(
            self.textgrid_to_table(
                x=x,
                tiername=tiername,
                filename=filename,
                columnname=columnname
            )
        )

    def table_to_textgrid(
            self,
            x: pd.DataFrame,
            tiername: str,
            columnname: str = "labels",
            tg: dict = None
    ) -> dict:
        
        assert "start" in x.columns, "DataFrame does not contain time info"
        assert columnname in x.columns, "DataFrame does not contain column"
        lab = x[columnname].to_list()
        if "end" not in x.columns:
            t = x["start"].to_numpy()
        else:
            t = np.column_stack(
                (x["start"].to_numpy(),
                 x["end"].to_numpy())
            )
        
        tgp = TextGridProc()
        tier = tgp.table2tier(t=t, lab=lab, specs={"name": tiername})
        return tgp.add_tier(tg=tg, tier=tier)
        
    def audformat_to_textgrid(
            self,
            x: pd.DataFrame,
            tiername: str,
            columnname: str = "labels",
            tg: dict = None
    ) -> dict:

        return self.table_to_textgrid(
            x=self.audformat_to_table(x),
            tiername=tiername,
            columnname=columnname,
            tg=tg
        )
        
    
class TextGridProc(object):

    # copy code from ~/github/copasul/copasul/copasul_utily.py

    def __init__(self):

        r"""TextGrid processing"""

        super().__init__()

        
    def read(self, f: str) -> dict:

        r""" read TextGrid file

        Args:
            f: (str) file name
    
        Returns:
            dict with keys
                type: TextGrid
                format: short|long
                name: s name of file
                head: nested dict xmin|xmax|size|type
                item_name -> myTiername -> myItemIdx
                item
                    myItemIdx -> (same as for item_name->myTiername)
                    class
                    name
                    size
                    xmin
                    xmax
                    intervals   if class=IntervalTier
                        myIdx -> (xmin|xmax|text)
                    points
                        myIdx -> (time|mark)
        """

        if self.which_format(f) == 'long':
            return self._read_long(f)
        else:
            return self._read_short(f)

        
    def which_format(self, f: str) -> str:

        r""" decides whether TextGrid is in long or short format
    
        Args:
            f: (str) TextGrid file name
    
        Returns:
            'short' or 'long'
        """

        with open(f, encoding='utf-8') as h:
            for z in h:
                if re.search(r'^\s*<exists>', z):
                    h.close
                    return 'short'
                elif re.search(r'xmin\s*=', z):
                    h.close
                    return 'long'
        return 'long'


    def _read_short(self, f: str) -> dict:

        r""" TextGrid short format input
    
        Args:
            f: (str) file name

        Returns:
            dict, see self.read()
        """

        tg = {'name': f, 'format': 'short', 'head': {},
              'item_name': {}, 'item': {}, 'type': 'TextGrid'}
        (key, fld, skip, state, nf) = ('head', 'xmin', True, 'head', self._next_field())
        idx = {'item': 0, 'points': 0, 'intervals': 0}
    
        with open(h, encoding='utf-8') as h:
            for z in h:
                z = re.sub(r'\s*\n$', '', z)
                if re.search(r'object\s*class', z, re.I):
                    fld = nf[state]['#']
                    skip = False
                    continue
                elif ((skip == True) or re.search(r'^\s*$', z) or
                      re.search('<exists>', z)):
                    continue
                if re.search(r'(interval|text)tier', z, re.I):
                    if re.search('intervaltier', z, re.I):
                        typ = 'interval'
                    else:
                        typ = 'text'
                    z = re.sub('"', '', z)
                    key = 'item'
                    state = 'item'
                    fld = nf[state]['#']
                    idx[key] += 1
                    idx['points'] = 0
                    idx['intervals'] = 0
                    if not (idx[key] in tg[key]):
                        tg[key][idx[key]] = {}
                    tg[key][idx[key]][fld] = z
                    if re.search('text', typ, re.I):
                        subkey = 'points'
                    else:
                        subkey = 'intervals'
                    fld = nf[state][fld]
                else:
                    z = re.sub('"', '', z)
                    if fld == 'size':
                        z = int(z)
                    elif fld in ['xmin', 'xmax', 'time']:
                        z = float(z)
                    if state == 'head':
                        tg[key][fld] = z
                        fld = nf[state][fld]
                    elif state == 'item':
                        tg[key] = add_subdict(tg[key], idx[key])
                        tg[key][idx[key]][fld] = z
                        if fld == 'name':
                            tg['item_name'][z] = idx[key]
                        # last fld of item reached
                        if nf[state][fld] == '#':
                            state = subkey
                            fld = nf[state]['#']
                        else:
                            fld = nf[state][fld]
                    elif re.search(r'(points|intervals)', state):
                        # increment points|intervals idx if first field adressed
                        if fld == nf[state]['#']:
                            idx[subkey] += 1
                        tg[key][idx[key]] = add_subdict(tg[key][idx[key]], subkey)
                        tg[key][idx[key]][subkey] = add_subdict(
                            tg[key][idx[key]][subkey], idx[subkey])
                        tg[key][idx[key]][subkey][idx[subkey]][fld] = z
                        if nf[state][fld] == '#':
                            fld = nf[state]['#']
                        else:
                            fld = nf[state][fld]
                        
        return tg
        

    def _read_long(self, f: str) -> dict:

        r""" TextGrid long format input
    
        Args:
            f: (str) file name
    
        Returns:
            dict; see self.read()
        """

        tg = {'name': f, 'format': 'long', 'head': {},
              'item_name': {}, 'item': {}}
        (key, skip) = ('head', True)
        idx = {'item': 0, 'points': 0, 'intervals': 0}
    
        with open(f, encoding='utf-8') as h:
            for z in h:
                z = re.sub(r'\s*\n$', '', z)
                if re.search(r'object\s*class', z, re.I):
                    skip = False
                    continue
                elif ((skip == True) or re.search(r'^\s*$', z) or
                      re.search('<exists>', z)):
                    continue
                if re.search(r'item\s*\[\s*\]:?', z, re.I):
                    key = 'item'
                elif re.search(r'(item|points|intervals)\s*\[(\d+)\]\s*:?', z, re.I):
                    m = re.search(
                        r'(?P<typ>(item|points|intervals))\s*\[(?P<idx>\d+)\]\s*:?', z)
                    i_type = m.group('typ').lower()
                    idx[i_type] = int(m.group('idx'))
                    if i_type == 'item':
                        idx['points'] = 0
                        idx['intervals'] = 0
                elif re.search(r'([^\s+]+)\s*=\s*\"?(.*)', z):
                    m = re.search(r'(?P<fld>[^\s+]+)\s*=\s*\"?(?P<val>.*)', z)
                    (fld, val) = (m.group('fld').lower(), m.group('val'))
                    fld = re.sub('number', 'time', fld)
                    val = re.sub(r'[\"\s]+$', '', val)
                    # type cast
                    if fld == 'size':
                        val = int(val)
                    elif fld in ['xmin', 'xmax', 'time']:
                        val = float(val)
                    # head specs
                    if key == 'head':
                        tg[key][fld] = val
                    else:
                        # link itemName to itemIdx
                        if fld == 'name':
                            tg['item_name'][val] = idx['item']
                        if ((idx['intervals'] == 0) and (idx['points'] == 0)):
                            # item specs
                            tg[key] = add_subdict(tg[key], idx['item'])
                            tg[key][idx['item']][fld] = val
                        else:
                            # points/intervals specs
                            tg[key] = add_subdict(tg[key], idx['item'])
                            tg[key][idx['item']] = add_subdict(
                                tg[key][idx['item']], i_type)
                            tg[key][idx['item']][i_type] = add_subdict(
                                tg[key][idx['item']][i_type], idx[i_type])
                            tg[key][idx['item']][i_type][idx[i_type]][fld] = val
                            
        return tg

    
    
    def _next_field(self) -> dict:

        r"""
        returns next TextGrid item dict

        Returns:
            dict mapping each item to the one following in TextGrid

        """

        return {'head':
                {'#': 'xmin',
                 'xmin': 'xmax',
                 'xmax': 'size',
                 'size': '#'},
                'item':
                {'#': 'class',
                 'class': 'name',
                 'name': 'xmin',
                 'xmin': 'xmax',
                 'xmax': 'size',
                 'size': '#'},
                'points':
                {'#': 'time',
                 'time': 'mark',
                 'mark': '#'},
                'intervals':
                {'#': 'xmin',
                 'xmin': 'xmax',
                 'xmax': 'text',
                 'text': '#'}}

    def tier(self, tg: str, tn: str) -> dict:
        
        r""" returns tier subdict from TextGrid
    
        Args:
            tg: (dict) by i_tg()
            tn: (str) name of tier
    
        Returns:
            tier (copied)
        """

        if tn not in tg['item_name']:
            return {}
    
        return copy.deepcopy(tg['item'][tg['item_name'][tn]])

    
    def tiernames(self, tg: dict) -> list:

        r""" returns list of TextGrid tier names
    
        Args:
            tg: (dict)
    
        Returns:
            tn: (list) sorted list of tiernames
        """

        return sorted(list(tg['item_name'].keys()))

    
    def tiertype(self, t: dict) -> str:

        r""" returns tier type
    
        Args:
            t: (dict) tg tier by self.tier()
    
        Returns:
            typ: (str) 'points'|'intervals'|''
        """

        for x in ['points', 'intervals']:
            if x in t:
                return x
        
        return ""

    
    def add_tier(self, tg: dict, tier: dict,
                 opt: dict = {'repl': True}) -> dict:

        r""" add tier to TextGrid
    
        Args:
            tg: (dict) from i_tg(); can be empty dict
            tier: (dict) subdict to be added:
                same dict form as in i_tg() output, below 'myItemIdx'
            opt: dict with keys
                'repl': - replace tier of same name
    
        Returns:
            tg: (dict) updated

        Comments:
            if generated from scratch head xmin and xmax are taken over from
            the tier - which might need to be corrected afterwards!
        """

        if tg is None:
            tg = {}
        
        # from scratch
        if 'item_name' not in tg:
            fromScratch = True
            tg = {'name': '', 'format': 'long', 'item_name': {}, 'item': {},
                  'head': {'size': 0, 'xmin': 0, 'xmax': 0, 'type': 'ooTextFile'}}
        else:
            fromScratch = False

        # tier already contained?
        if (opt['repl'] == True and (tier['name'] in tg['item_name'])):
            i = tg['item_name'][tier['name']]
            tg['item'][i] = tier
        else:
            # item index
            ii = sorted(tg['item'].keys())
            if len(ii) == 0:
                i = 1
            else:
                i = ii[-1]+1
            tg['item_name'][tier['name']] = i
            tg['item'][i] = tier
            tg['head']['size'] += 1

        if fromScratch and 'xmin' in tier:
            for x in ['xmin', 'xmax']:
                tg['head'][x] = tier[x]
                
        return tg


    def table2tier(self, t: np.array,
                   lab: list, specs: dict) -> dict:

        r""" transforms table to TextGrid tier
    
        Args:
            t: (np.array) 1- or 2-dim array with time info
            lab: (list) of labels
            specs: dict with keys
                ['class'] <'IntervalTier' for 2-dim, 'TextTier' for 1-dim>
                ['name'] required key
                ['xmin'] <0>
                ['xmax'] <max tab>
                ['size'] - will be determined automatically
                ['lab_pau'] - <''>
    
        Returns:
            dict tg tier (see i_tg() subdict below myItemIdx)
    
        Comments:
            for 'interval' tiers gaps between subsequent intervals will
            be bridged by lab_pau
        """

        tt = {'name': specs['name']}
        nd = t.ndim

        # 2dim array with 1 col
        if nd == 2:
            nd = t.shape[1]

        # tier class
        if nd == 1:
            tt['class'] = 'TextTier'
            tt['points'] = {}
        else:
            tt['class'] = 'IntervalTier'
            tt['intervals'] = {}

            # pause label for gaps between intervals
            if 'lab_pau' in specs:
                lp = specs['lab_pau']
            else:
                lp = ''

        # xmin, xmax
        if 'xmin' not in specs:
            tt['xmin'] = 0
        else:
            tt['xmin'] = specs['xmin']
        if 'xmax' not in specs:
            if nd == 1:
                tt['xmax'] = t[-1]
            else:
                tt['xmax'] = t[-1, 1]
        else:
            tt['xmax'] = specs['xmax']
    
        if nd == 1:
            # point tier content
            for i in np.arange(0, len(t), 1):

                # point tier content might be read as
                # [[x],[x],[x],...] or [x,x,x,...]
                if type(t[i]) in [list, np.array]:
                    z = t[i, 0]
                else:
                    z = t[i]
                tt['points'][i + 1] = {'time': z, 'mark': lab[i]}
            tt['size'] = len(t)
        else:
            # interval tier content
            j = 1
        
            # initial pause
            if t[0, 0] > tt['xmin']:
                tt['intervals'][j] = {'xmin': tt['xmin'],
                                      'xmax': t[0, 0], 'text': lp}
                j += 1
            for i in np.arange(0, len(t), 1):

                # pause insertions
                if ((j - 1 in tt['intervals']) and
                    t[i, 0] > tt['intervals'][j - 1]['xmax']):
                    tt['intervals'][j] = {'xmin': tt['intervals'][j-1]['xmax'],
                                          'xmax': t[i, 0], 'text': lp}
                    j += 1
                tt['intervals'][j] = {'xmin': t[i, 0],
                                      'xmax': t[i, 1], 'text': lab[i]}
                j += 1

            # final pause
            if tt['intervals'][j-1]['xmax'] < tt['xmax']:
                tt['intervals'][j] = {'xmin': tt['intervals'][j-1]['xmax'],
                                      'xmax': tt['xmax'], 'text': lp}
                j += 1  # uniform 1 subtraction for size

            # size
            tt['size'] = j - 1
        
        return tt

    
    def tier2table(self, t: dict, opt: dict=None) -> Tuple[np.array, list]:

        r"""
        transforms TextGrid tier to 2 arrays
        point -> 1 dim + lab
        interval -> 2 dim (one row per segment) + lab
    
        Args:
            t: (dict) tg tier (by self.tier())
            opt: (dict) with keys
               "skip": list of labels to be skipped 
               "skip_empty": boolean, if True, empty strings will be skippes
    
        Returns:
            x: (np.array) 1- or 2-dim array of time stamps
            lab: (list) corresponding labels

        Comments:
            empty intervals are skipped
        """

        opt = opt_default(opt, {"skip": [], "skip_empty": True})
        if len(opt["skip"]) > 0:
            do_skip = True
        else:
            do_skip = False
            
        x, lab = [], []
    
        if 'intervals' in t:
            for i in sorted(t['intervals'].keys()):
                z = t['intervals'][i]
                if opt["skip_empty"] and len(z['text']) == 0:
                    continue
                if z["text"] in opt["skip"]:
                    continue

                x.append([z['xmin'], z['xmax']])
                lab.append(z['text'])
        else:
            for i in sorted(t['points'].keys()):
                z = t['points'][i]
                if do_skip and re.search(opt["skip"], z["mark"]):
                    continue
                x.append(z['time'])
                lab.append(z['mark'])

        x = np.array(x)
            
        return x, lab

    
    def _fieldnames(self) -> dict:

        r"""
        returns field names of TextGrid head and items
    
        Returns:
            TextGrid element name -> list of field names
        """

        return {'head': ['xmin', 'xmax', 'size'],
                'item': ['class', 'name', 'xmin', 'xmax', 'size'],
                'points': ['time', 'mark'],
                'intervals': ['xmin', 'xmax', 'text']}

    
    def write(self, tg: dict, fil: str) -> None:

        r""" writes TextGrid to file
        (content is appended if file exists)
    
        Args:
            tg: (dict) read by i_tg()
            fil: (str) output file name
    
        """

        h = open(fil, mode='w', encoding='utf-8')
        idt = '    '
        fld = self._fieldnames()

        # head
        if tg['format'] == 'long':
            h.write("File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n")
            h.write("xmin = {}\n".format(self._tgv(tg['head']['xmin'], 'xmin')))
            h.write("xmax = {}\n".format(self._tgv(tg['head']['xmax'], 'xmax')))
            h.write("tiers? <exists>\n")
            h.write("size = {}\n".format(self._tgv(tg['head']['size'], 'size')))
        else:
            h.write("File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n")
            h.write("{}\n".format(self._tgv(tg['head']['xmin'], 'xmin')))
            h.write("{}\n".format(self._tgv(tg['head']['xmax'], 'xmax')))
            h.write("<exists>\n")
            h.write("{}\n".format(self._tgv(tg['head']['size'], 'size')))

        # item
        if (tg['format'] == 'long'):
            h.write("item []:\n")

        for i in sorted(tg['item'].keys()):

            # subkey := intervals or points?
            if re.search(tg['item'][i]['class'], 'texttier', re.I):
                subkey = 'points'
            else:
                subkey = 'intervals'

            if tg['format'] == 'long':
                h.write(f"{idt}item [{i}]:\n")

            for f in fld['item']:
                if tg['format'] == 'long':
                    if f == 'size':
                        h.write("{}{}{}: size = {}\n".format(
                            idt, idt, subkey,
                            self._tgv(tg['item'][i]['size'], 'size')))
                    else:
                        h.write("{}{}{} = {}\n".format(
                            idt, idt, f, self._tgv(tg['item'][i][f], f)))
                else:
                    h.write("{}\n".format(self._tgv(tg['item'][i][f], f)))

            # empty tier
            if subkey not in tg['item'][i]:
                continue

            for j in sorted(tg['item'][i][subkey].keys()):
                if tg['format'] == 'long':
                    h.write(f"{idt}{idt}{subkey} [{j}]:\n")
                for f in fld[subkey]:
                    if (tg['format'] == 'long'):
                        h.write("{}{}{}{} = {}\n".format(
                            idt, idt, idt,
                            f, self._tgv(tg['item'][i][subkey][j][f], f)))
                    else:
                        h.write("{}\n".format(
                            self._tgv(tg['item'][i][subkey][j][f], f)))

        h.close()

        
    def _tgv(self, v: Any, a: str) -> Any:

        r""" rendering of TextGrid values
    
        Args:
            v: value
            a: name of attribute
    
        Returns:
            s: rendered value
        """

        if re.search(r'(xmin|xmax|time|size)', a):
            return v
        else:
            return f"\"{v}\""

        
def logits2probs(
        x: Union[np.array, pd.DataFrame]
) -> Union[np.array, pd.DataFrame]:

    r""" converts logits in 2-dim np.array x to probs

    Args:
        logits: input logits

    Returns:
        probabilities

    Remarks:
        variable is changed inplace

    """
    
    odds = np.exp(x)
    nrm = np.sum(odds, axis=1)
    nrm[nrm==0] = 0.0000000000000001
    nrms = np.column_stack((nrm,nrm))
    while nrms.shape[1] < odds.shape[1]:
        nrms = np.column_stack((nrms,nrm))
    x = odds / nrms
    
    return x


def logits2labels(
        x: Union[np.array, pd.DataFrame],
        idx2label: dict = None
) -> Union[np.array, pd.DataFrame]:

    """ converts logits to argmax indices 
    and maps these to labels 

    Args:
        x: logits
        idx2label: dict mapping column indices to label strings
            (used for array input only)

    Returns:
        lab: labels
        
    Remarks:
        If x is array:
            - output is array of labels
            - labels are values from idx2label if provided, else
                column indices
        If x is DataFrame:
            - output is dataframe with single column "labels" 
            - labels are column names from input DataFrame
    """

    if type(x) is pd.DataFrame:
        u = x.to_numpy()
        i2l = {}
        for i, col in enumerate(x.columns):
            i2l[i] = col
    else:
        u = x
        i2l = idx2label
        
    lab = np.argmax(u, axis=1)

    # array input, no label mapping
    if i2l is None:
        return lab

    # label mapping
    lab = lab.tolist()
    for i in range(len(lab)):
        if lab[i] in i2l:
            lab[i] = i2l[lab[i]]

    # return same type
    lab = np.array(lab)
    if type(x) is pd.DataFrame:
        return pd.DataFrame({"labels": lab}, index=x.index)
    
    return lab


def add_subdict(d, s):

    r"""
    add key to empty subdict if not yet part of dict
    
    Args:
        d: (dict)
        s: (key)
    
    Returns:
        d: (dict) incl key pointing to empty subdict
    """

    if s not in d:
        d[s] = {}
        
    return d


def opt_default(c: dict, d: dict) -> dict:

    r""" recursively adds default fields of dict d to dict c
    if not yet specified in c
    
    Args:
       c: (dict) some dict
       d: (dict) dict with default values
    
    Returns:
       c: (dict) merged dict (defaults added to c)
    """

    if c is None:
        c = {}
    
    for x in d:
        if x not in c:
            c[x] = d[x]
        elif type(c[x]) is dict:
            c[x] = opt_default(c[x], d[x])
            
    return c
