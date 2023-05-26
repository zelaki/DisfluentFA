#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import re
from praatio import textgrid
from itertools import groupby
from librosa.sequence import dtw
from dataclasses import dataclass
import matplotlib
import matplotlib.pyplot as plt

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start



def ctc2duration(phones,resolution=0.01):
    """
    xxxxx convert ctc to duration

    Parameters
    ----------
    phones : list
        A list of phone sequence
    resolution : float, optional
        The resolution of xxxxx. The default is 0.01.

    Returns
    -------
    merged : list
        xxxxx A list of duration values.

    """
    
    counter = 0
    out = []
    for p,group in groupby(phones):
        length = len(list(group))
        out.append((round(counter*resolution,2),round((counter+length)*resolution,2),p))
        counter += length
        
    merged = []
    for i, (s,e,p) in enumerate(out):
        if i==0 and p=='[PAD]':
            merged.append((s,e,'[SIL]'))
        elif p=='[PAD]':
            merged.append((out[i-1][0],e,out[i-1][2]))
        elif i==len(out)-1:
            merged.append((s,e,p))
    return merged


def seq2duration(phones,resolution=0.01):
    """
    xxxxx convert phone sequence to duration

    Parameters
    ----------
    phones : list
        A list of phone sequence
    resolution : float, optional
        The resolution of xxxxx. The default is 0.01.

    Returns
    -------
    out : list
        xxxxx A list of duration values.

    """
    
    counter = 0
    out = []
    for p,group in groupby(phones):
        length = len(list(group))
        out.append((round(counter*resolution,2),round((counter+length)*resolution,2),p))
        counter += length
    return out


def duration2textgrid(duration_seq,save_path=None):
    """
    Save duration values to textgrids

    Parameters
    ----------
    duration_seq : list
        xxxxx A list of duration values.
    save_path : str, optional
        The path to save the TextGrid files. The default is None.

    Returns
    -------
    tg : TextGrid file?? str?? xxxxx?
        A textgrid object containing duration information.

    """

    tg = textgrid.Textgrid()
    phoneTier = textgrid.IntervalTier('phones', duration_seq, 0, duration_seq[-1][1])
    tg.addTier(phoneTier)
    if save_path:
        tg.save(save_path,format="short_textgrid", includeBlankSpaces=False)
    return tg


def word2textgrid(duration_seq,word_seq,save_path=None):
    """
    Save duration values to textgrids

    Parameters
    ----------
    duration_seq : list
        xxxxx A list of duration values.
    save_path : str, optional
        The path to save the TextGrid files. The default is None.

    Returns
    -------
    tg : TextGrid file?? str?? xxxxx?
        A textgrid object containing duration information.

    """

    tg = textgrid.Textgrid()
    phoneTier = textgrid.IntervalTier('phones', duration_seq, 0, duration_seq[-1][1])
    tg.addTier(phoneTier)
    # wordTier = textgrid.IntervalTier('words', word_seq, 0, word_seq[-1][1])
    # tg.addTier(wordTier)
    if save_path:
        tg.save(save_path,format="short_textgrid", includeBlankSpaces=False)
    return tg



def get_boundaries(phone_seq):
    """
    Get time of phone boundaries

    Parameters
    ----------
    phone_seq : list xxxx?
        A list of phone sequence.

    Returns
    -------
    timings: A list of time stamps
    symbols: A list of phone symbols

    """
    
    boundaries = defaultdict(set)
    for s,e,p in phone_seq:
        boundaries[s].update([p.upper()])
#        boundaries[e].update([p.upper()+'_e'])
    timings = np.array(list(boundaries.keys()))
    symbols = list(boundaries.values())
    return (timings,symbols)


def check_textgrid_duration(textgrid,duration):
    """
    Check whether the duration of a textgrid file equals to 'duration'. 
    If not, replace duration of the textgrid file.

    Parameters
    ----------
    textgrid : .TextGrid object
        A .TextGrid object.
    duration : float
        A given length of time.

    Returns
    -------
    textgrid : .TextGrid object
        A modified/unmodified textgrid.

    """
    
    
    endtime = textgrid.tierDict['phones'].entryList[-1].end
    if not endtime==duration:
        last = textgrid.tierDict['phones'].entryList.pop()
        textgrid.tierDict['phones'].entryList.append(last._replace(end=duration))
        
    return textgrid
    

def textgrid_to_labels(phones,duration,resolution):
    """
    

    Parameters
    ----------
    phones : list
        A list of phone sequence
    resolution : float, optional
        The resolution of xxxxx. The default is 0.01.
    duration : float
        A given length of time.
    

    Returns
    -------
    labels : list
        A list of phone labels.

    """
    
    labels = []
    clock = 0.0

    for i, (s,e,p) in enumerate(phones):

        assert clock >= s
        while clock <= e:
            labels.append(p)
            clock += resolution
        
        # if more than half of the current frame is outside the current phone
        # we'll label it as the next phone
        if np.abs(clock-e) > resolution/2:
            labels[-1] = phones[min(len(phones)-1,i+1)][2]
    
    # if the final time interval is longer than the total duration
    # we will chop off this frame
    if clock-duration > resolution/2:
        labels.pop()

    return labels

def remove_null_and_numbers(labels):
    """
    Remove labels which are null, noise, or numbers.

    Parameters
    ----------
    labels : list
        A list of text labels.

    Returns
    -------
    out : list
        A list of new labels.

    """
    
    out = []
    noises = set(['SPN','NSN','LAU'])
    for l in labels:
        l = re.sub(r'\d+','',l)
        l = l.upper()
        if l == '' or l == 'SIL':
            l = '[SIL]'
        if l == 'SP':
            l = '[SIL]'
        if l in noises:
            l = '[UNK]'
        out.append(l)
    return out


def insert_sil(phones):
    """
    Insert silences.

    Parameters
    ----------
    phones : list
        A list of phone sequence

    Returns
    -------
    out : list
        A list of new labels.

    """
    
    out = []
    for i,(s,e,p) in enumerate(phones):
        
        if out:
            if out[-1][1]!=s:
                out.append((out[-1][1],s,'[SIL]'))
        out.append((s,e,p))
    return out

def forced_align(cost, phone_ids):

    """
    Force align text to audio.

    Parameters
    ----------
    cost : float xxxxx
        xxxxx.
    phone_ids : list
        A list of phone IDs.

    Returns
    -------
    align_id : list
        A list of IDs for aligned phones.

    """

    D,align = dtw(C=-cost[:,phone_ids],
                  step_sizes_sigma=np.array([[1, 1], [1, 0]]))
    # trellis = get_trellis(cost, phone_ids)
    # path = backtrack(trellis, torch.tensor(cost), phone_ids)
    # plot_trellis_with_path(trellis, path)
    # print(align)
    # print(path)
    # plt.imshow(trellis.T)
    # plt.colorbar()
    # plt.savefig('trellis.png')

    # plt.imshow(cost[:,phone_ids].T)
    # plt.colorbar()
    # plt.savefig('emission_phones.png')
    # plt.imshow(-D.T)
    # plt.colorbar()
    # plt.savefig('cost.png')


    align_seq = [-1 for i in range(max(align[:,0])+1)]
    for i in list(align):
    #    print(align)
        if align_seq[i[0]]<i[1]:
            align_seq[i[0]]=i[1]

    align_id = list(align_seq)
    return align_id



def get_trellis(emission, tokens, blank_id=0):
    emission = torch.tensor(emission)
    num_frame = emission.shape[0]
    num_tokens = len(tokens)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")
    
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, tokens],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis




def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]

def plot_trellis_with_path(trellis, path, image_path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for _, p in enumerate(path):
        # print(p.time_index, p.token_index)
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_under('red')
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow(trellis_with_path[1:, 1:].T, origin="lower",cmap=current_cmap, aspect='auto')
    plt.tight_layout()

    plt.savefig(image_path)





def plot_trellis_with_segments(trellis, segments, transcript, path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

    fig, ax1 = plt.subplots(figsize=(16, 9.5))
    ax1.set_title("Path, label and probability for each label")
    ax1.imshow(trellis_with_path.T, origin="lower")
    ax1.set_xticks([])

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start + 0.7, i + 0.3), weight="bold")
            ax1.annotate(f"{seg.score:.2f}", (seg.start - 0.3, i + 4.3))

    # ax2.set_title("Label probability with and without repetation")
    # xs, hs, ws = [], [], []
    # for seg in segments:
    #     if seg.label != "|":
    #         xs.append((seg.end + seg.start) / 2 + 0.4)
    #         hs.append(seg.score)
    #         ws.append(seg.end - seg.start)
    #         ax2.annotate(seg.label, (seg.start + 0.8, -0.07), weight="bold")
    # ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

    # xs, hs = [], []
    # for p in path:
    #     label = transcript[p.token_index]
    #     if label != "|":
    #         xs.append(p.time_index + 1)
    #         hs.append(p.score)

    # ax2.bar(xs, hs, width=0.5, alpha=0.5)
    # ax2.axhline(0, color="black")
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_ylim(-0.1, 1.1)






if __name__ == '__main__':
    '''
    Testing functions
    '''    

    pass 











