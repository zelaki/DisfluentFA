from src.Charsiu import charsiu_forced_aligner
import sys
import os
from praatio import textgrid


def write_textgrid(phone_alignment, grid_path):
    tg = textgrid.Textgrid()
    phoneTier = textgrid.IntervalTier('phones', phone_alignment, 0, phone_alignment[-1][1])
    tg.addTier(phoneTier)
    tg.save(grid_path,format="short_textgrid", includeBlankSpaces=False)


def read_text(word_path):
    with open(word_path) as fd:
        return fd.readline()


audio_path = sys.argv[1]
word_path = sys.argv[2]


charsiu = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')

text = read_text(word_path)
phone_alignment = charsiu.wsfa(
    audio=audio_path,
    text = text,
    T=0.9
)


write_textgrid(phone_alignment, os.path.basename(audio_path)[:-8]+"_mod.textGrid")

print(phone_alignment)
