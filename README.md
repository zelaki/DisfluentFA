## Forced Alignment for Disfluent Speech 
#### Code for: [Weakly-supervised forced alignment of disfluent speech using phoneme-level modeling](https://www.isca-speech.org/archive/pdfs/interspeech_2023/kouzelis23_interspeech.pdf)
Presented at ISCA Interspeech 2023.
## Project Status
This repository is currently under development and is a work in progress. Contributions and feedback are welcome!






### Intro
The study of speech disorders can benefit greatly from
time-aligned data. However, audio-text mismatches in disfluent
speech cause rapid performance degradation for modern speech
aligners, hindering the use of automatic approaches. In this
work, we propose a simple and effective modification of align-
ment graph construction of CTC-based models using Weighted
Finite State Transducers. The proposed weakly-supervised ap-
proach alleviates the need for verbatim transcription of speech
disfluencies for forced alignment. During the graph construc-
tion, we allow the modeling of common speech disfluencies,
i.e. repetitions and omissions. 

<img src="local/modified_fsa.png" 
     width="450" 
     height="150" />



### Contents
- Code for weakly-supervised forced alignment of disfluent speech. The models and code for the frame classification model are based on [Charsiu](https://github.com/lingjzhu/charsiu)
- Code for the construction of **DisfluenTIMIT**, a corrupted version of the TIMIT test set with synthesized disfluencies
	
### Usage
```
Wealky-Supervised Forced Alignment

optional arguments:
  -a AUDIO, --audio AUDIO 	Path to the audio file.
  -t TEXT, --text TEXT  	Path to the text file.
  -w, --write_textgrid  	Specify whether to write a TextGrid file.
```
