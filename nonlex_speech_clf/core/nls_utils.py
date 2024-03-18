import audformat
import numpy as np
import pandas as pd
from typing import Any, Union

class FormatConverter(object):

    def __init__(
            self,
            source_format: str = "table",
            target_format: str = "audformat"
    ):

        r"""Format conversions

        Args:
           source_format: "table", "audformat", ("TextGrid")
           target_format: "table", "audformat", ("TextGrid")
        
        Returns:
           converted output

        Formats:
           "table": pd.DataFrame with columns "file" (file-level processing)
                or "file", "start", "end" (segment-level processing).
                Time one- and offsets in "start" and "end" provided in seconds
           "audformat": pd.Index "file" (file-level processing)
                or pd.MultiIndex "file", "start", "end" (segment-level processing).
                Time values are in timedelta format.
                pd.DataFrame with one of the two indices described above
           "textgrid": textgrid dict
        """

        super().__init__()
        
        self.source_format = source_format
        self.target_format = target_format
        
    def convert(
            self,
            x: Union[dict, pd.DataFrame, pd.Index, pd.MultiIndex],
            tg: dict = None,
            tier: str = None,
            channel: int = 0,
    ) -> Union[dict, pd.DataFrame]:

        """ conversions
        
        Args:
            x: input dict, dataframe, or index
            tg: textgrid dict (in case table should be added as new tier
                to exisiting TextGrid
            tier: tier name (required for all conversions from or to TextGrids)
            channel: channel index (for dataframe to TextGrid
                conversions; if input dataframe contains column
                "channel", only those rows with corresponding channel number
                are converted

        Returns:
            dataframe or Textgrid dict

        """
        
        if self.source_format == "audformat":
            if self.target_format == "table":
                return self._audformat_to_table(x)
            elif self.target_format == "textgrid":
                return self._audformat_to_textgrid(x, tg)
        elif self.source_format == "table":
            if self.target_format == "audformat":
                return self._table_to_audformat(x)
            elif self.target_format == "textgrid":
                return self._table_to_textgrid(x, tg)
        elif self.source_format == "textgrid":
            if self.target_format == "audformat":
                return self._textgrid_to_audformat(x)
            elif self.target_format == "table":
                return self._textgrid_to_table(x)
            
        return x


    def _table_to_audformat(
            self,
            x: pd.DataFrame
    ) -> pd.DataFrame:

        r""" converts dataframe to audformat; returns pd.DataFrame """
        
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

    
    def _audformat_to_table(
            self,
            x: Union[pd.DataFrame, pd.Index, pd.MultiIndex]
    ) -> pd.DataFrame:

        r""" converts audformat to table; returns DataFrame """

        if type(x) is pd.DataFrame:
            index = x.index
        else:
            index = x
        
        files = index.get_level_values("file")
        starts, ends = None, None
        if audformat.is_segmented_index(index):    
            starts = index.get_level_values("start").total_seconds().to_numpy()
            ends = index.get_level_values("start").total_seconds().to_numpy()
        x.reset_index(inplace=True)
        x["file"] = files
        if starts is not None:
            x["start"] = starts
            x["end"] = ends

        return x


    def _textgrid_to_table(
            self,
            x: dict,
            tier: str
    ) -> pd.DataFrame:
        pass

    def _textgrid_to_audformat(
            self,
            x: dict,
            tier: str
    ) -> pd.DataFrame:
        pass

    def _table_to_textgrid(
            self,
            x: pd.DataFrame,
            tg: dict = None
    ) -> dict:
        pass

    def _audformat_to_textgrid(
            self,
            x: pd.DataFrame,
            tg: dict = None
    ) -> dict:
        pass
    
        
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

