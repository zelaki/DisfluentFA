#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from icefall.utils import (
    get_alignments,
)
import re
from icefall.decode import one_best_decoding, nbest_decoding, Nbest
import matplotlib.pyplot as plt
import numpy as np

import sys
import torch
from itertools import groupby
sys.path.append('src/')
import numpy as np
from models import  Wav2Vec2ForFrameClassification
from utils import seq2duration,duration2textgrid
from processors import CharsiuPreprocessor_zh, CharsiuPreprocessor_en, CharsiuPreprocessor_nb
import k2
from modified_liner import get_modified_fsa
from src.topologies import *
processors = {'zh':CharsiuPreprocessor_zh,
              'en':CharsiuPreprocessor_en,
              'nb':CharsiuPreprocessor_nb}

class charsiu_aligner:
    
    def __init__(self, 
                 lang='en', 
                 sampling_rate=16000, 
                 device=None,
                 recognizer=None,
                 processor=None, 
                 resolution=0.01):
                
        self.lang = lang 
        
        if processor is not None:
            self.processor = processor
        else:
            self.charsiu_processor = processors[self.lang]()
        
        
        
        self.resolution = resolution
        
        self.sr = sampling_rate
        
        self.recognizer = recognizer
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    
    def _freeze_model(self):
        self.aligner.eval().to(self.device)
        if self.recognizer is not None:
            self.recognizer.eval().to(self.device)
    
    
    
    def align(self,audio,text):
        raise NotImplementedError()
        
        
        
    def serve(self,audio,save_to,output_format='variable',text=None):
        raise NotImplementedError()
        
    
    def _to_textgrid(self,phones,save_to):
        '''
        Convert output tuples to a textgrid file

        Parameters
        ----------
        phones : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        duration2textgrid(phones,save_path=save_to)
        print('Alignment output has been saved to %s'%(save_to))
    
    
    
    def _to_tsv(self,phones,save_to):
        '''
        Convert output tuples to a tab-separated file

        Parameters
        ----------
        phones : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        with open(save_to,'w') as f:
            for start,end,phone in phones:
                f.write('%s\t%s\t%s\n'%(start,end,phone))
        print('Alignment output has been saved to %s'%(save_to))






class charsiu_forced_aligner(charsiu_aligner):
    
    def __init__(self, aligner, sil_threshold=4, **kwargs):
        super(charsiu_forced_aligner, self).__init__(**kwargs)
        self.aligner = Wav2Vec2ForFrameClassification.from_pretrained(aligner)
        self.sil_threshold = sil_threshold
        
        self._freeze_model()
        
    def wsfa(self, audio, text, T=0.9, ws=True):
        '''
        Perform forced alignment

        Parameters
        ----------
        audio : np.ndarray [shape=(n,)]
            time series of speech signal
        text : str
            The transcription

        Returns
        -------
        A tuple of aligned phones in the form (start_time, end_time, phone)

        '''
        audio_path = audio
        audio = self.charsiu_processor.audio_preprocess(audio,sr=self.sr)
        audio = torch.Tensor(audio).unsqueeze(0).to(self.device)
        phones, words = self.charsiu_processor.get_phones_and_words(text)
        phones_ids_grouped = [[self.charsiu_processor.mapping_phone2id(ph) for ph in list(word)] for word in phones]
        start_word_indexes = list(np.cumsum([len(word) for word in phones_ids_grouped]))
        phone_ids = self.charsiu_processor.get_phone_ids(phones)
        with torch.no_grad():
            out = self.aligner(audio)
        cost = torch.softmax(out.logits,dim=-1).detach().cpu().numpy().squeeze()
        emission_k2 = out.logits.cpu().detach()
        supervision_segments = torch.tensor([[0,0,emission_k2.shape[1]]], dtype=torch.int32)

        dense_fsa_vec = k2.DenseFsaVec(
            emission_k2,
            supervision_segments,
            allow_truncate=0,
        )

        if ws:
            linear_fsa = get_modified_fsa(phone_ids, start_word_indexes, T)

        else:
            linear_fsa = k2.linear_fsa([phone_ids])

        ctc_topo = k2.ctc_topo(max_token=42, modified=True)
        decoding_graph = k2.compose(ctc_topo, linear_fsa)
        lattice = k2.intersect_dense(
            decoding_graph,
            dense_fsa_vec,
            20,
        )
        best_path = one_best_decoding(
            lattice=lattice,
            use_double_scores=True,
        )
        phones = [item for sublist in phones for item in sublist]
        phones = [re.sub(r'\d','',p) for p in phones]


        labels_ali = get_alignments(best_path, kind="labels")
        phone_ali = [self.charsiu_processor.mapping_id2phone(ph) for ph in labels_ali[0]]
        phone_ali =  seq2duration(phone_ali,resolution=self.resolution)
 
        
        return phone_ali




    def wsfa_long(self, audio, text, T=0.9, ws=True):
        '''
        Perform forced alignment

        Parameters
        ----------
        audio : np.ndarray [shape=(n,)]
            time series of speech signal
        text : str
            The transcription

        Returns
        -------
        A tuple of aligned phones in the form (start_time, end_time, phone)

        '''

        audio = self.charsiu_processor.long_audio_process(audio,sr=self.sr)
        audio = [torch.Tensor(seg).unsqueeze(0).to(self.device) for seg in audio]
        phones, words = self.charsiu_processor.get_phones_and_words(text)
        phones_ids_grouped = [[self.charsiu_processor.mapping_phone2id(ph) for ph in list(word)] for word in phones]

        start_word_indexes = list(np.cumsum([len(word) for word in phones_ids_grouped]))
        phone_ids = self.charsiu_processor.get_phone_ids(phones)
        logits = []
        with torch.no_grad():
            for seg in audio:
                out = self.aligner(seg)
                logits.append(out.logits)
        logits = torch.cat(tuple(logits), 1)
        emission_k2 = logits.cpu().detach()

        supervision_segments = torch.tensor([[0,0,emission_k2.shape[1]]], dtype=torch.int32)

        dense_fsa_vec = k2.DenseFsaVec(
            emission_k2,
            supervision_segments,
            allow_truncate=0,
        )


        if ws:
            linear_fsa = get_modified_fsa(phone_ids, start_word_indexes, T)

        else:
            linear_fsa = k2.linear_fsa([phone_ids])


        ctc_topo = k2.ctc_topo(max_token=42, modified=True)
        decoding_graph = k2.compose(ctc_topo, linear_fsa)
        lattice = k2.intersect_dense(
            decoding_graph,
            dense_fsa_vec,
            20,
        )
        best_path = one_best_decoding(
            lattice=lattice,
            use_double_scores=True,
        )
        labels_ali = get_alignments(best_path, kind="labels")
        phone_ali = [self.charsiu_processor.mapping_id2phone(ph) for ph in labels_ali[0]]
        phone_ali =  seq2duration(phone_ali,resolution=self.resolution)


        return phone_ali
