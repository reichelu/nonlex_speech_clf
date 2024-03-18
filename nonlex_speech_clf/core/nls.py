import audonnx
import audinterface
import numpy as np
import pandas as pd
import torch
from typing import Union
import nonlex_speech_clf.core.nls_utils
import os

class NlsClf(object):

    def __init__(self, model_root: str = None):

        r"""Classifies speech segments into CG and FP

        Args:
            model_root: root directory for onnx model
        """

        super().__init__()

        if model_root is None:
            cwd = os.path.dirname(os.path.abspath(__file__))
            model_root= os.path.abspath(
                os.path.join(cwd, "..", "..", "model")
            )
            
        self.model = audonnx.load(model_root)

        
    def process_signal(self,
                       signal: np.array,
                       sampling_rate: int,
                       file: str = None,
                       start: float = None,
                       end: float = None,
                       channel: int = 0,
                       return_type: str = "logits") -> pd.DataFrame:
        
        r""" process signal

        Args:
            signal: mono signal array
            sampling_rate: sampling rate
            file: file path if to be added to output
                dataframe's index
            start: start point within signal in seconds
            end: end point withon signal in seconds
            channel: for stereo files - which channel (0 or 1)
            return_type:
                "logits": dataframe with logits for classes cg and fp
                "probabilities": dataframe with probs for cg and fp
                "classes": dataframe with single column "ans" and
                    values "cg" and "fp"

        Returns:
            dataframe with answers
        
        """
        
        signal = torch.from_numpy(signal).type(torch.float32)
        interface = self._init_interface(channel=channel)
        y = interface.process_signal(
            signal, sampling_rate, file=file,
            start=start, end=end)
        return self._return(y, return_type)
        

    def process_file(self,
                     file: str,
                     start: float = None,
                     end: float = None,
                     channel: int = 0,
                     return_type: str = "logits") -> pd.DataFrame:

        r""" process file

        Args:
            file: file name
            start: start point within signal in seconds
            end: end point withon signal in seconds
            channel: for stereo files - which channel (0 or 1)

        Returns:
            dataframe with answers

        """

        interface = self._init_interface(channel=channel)
        y = interface.process_file(file, start=start, end=end)
        return self._return(y, return_type)

    
    def process_index(self,
                      index: Union[pd.Index, pd.MultiIndex],
                      channel: int = 0,
                      num_jobs: int = 3,
                      return_type: str = "logits") -> pd.DataFrame:

        r""" processes index in audformat

        Args:
            index: index in audformat to be processed. Either of
                file or segmented type.
            channel: for stereo files - which channel (0 or 1)
            num_jobs: number of jobs for parallel processing
            return_type:
                "logits": dataframe with logits for classes cg and fp
                "probabilities": dataframe with probs for cg and fp
                "classes": dataframe with single column "ans" and
                    values "cg" and "fp"

        Returns:
            dataframe with answers
        
        """

        interface = self._init_interface(channel=channel, num_jobs=num_jobs)
        y = interface.process_index(index)
        return self._return(y, return_type)

    
    def _init_interface(
            self,
            channel: int = 0,
            num_jobs: int = 1
    ) -> object:

        """ initialize interface """

        return audinterface.Feature(
            self.model.outputs["logits"].labels,
            process_func=self.model,
            process_func_args={'outputs': "logits"},
            channels=channel,
            num_workers=num_jobs,
            verbose=True
        )
        
    
    def _return(self,
                y: pd.DataFrame,
                return_type: str = "logits") -> pd.DataFrame:

        """ wrapper around method returns """

        if return_type == "logits":
            return y

        if return_type == "probabilities":
            return logits2probs(y)
        
        return logits2class(y)
        
