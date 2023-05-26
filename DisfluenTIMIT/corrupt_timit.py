import sys
import os
import csv
from typing import List
from pydub import AudioSegment
from dataclasses import dataclass
import random
from praatio import textgrid
from syllabify.syllable3 import generate
import argparse


random.seed(42)


def parse_arguments():
	parser = argparse.ArgumentParser(
	"This script creats DisfuenTIMIT a disfluency augmented version of TIMIT test set",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
	"--words_dir",
	help="Path to directory of words files."
	)
	parser.add_argument(
	"--phones_dir",
	help="Path to directory of phones files."
	)
	parser.add_argument(
	"--wavs_dir",
	help="Path to directory of wav files."
	)
	parser.add_argument(
	"--output_dir",
	help="Path to DisfluenTIMIT dir"
	)
	return parser.parse_args()

@dataclass
class interval():
	onset: int
	offset: int
	label: str

@dataclass
class interval_corrupted:
	word: List[interval]
	label: int

@dataclass
class AudioSegmentsInfo:
	onset: float
	offset: float
	corr_type: int
	


def get_intervals(path):
    """
    Reads a text file with lines containing "onset offset word/phone" and returns a list of Intervals.
    
    Args:
        path (str): The path to the text file.
        
    Returns:
        list: A list of Interval objects created from the file's contents.
    """
    with open(path) as fd:
        lines = [ln.rstrip().split() for ln in fd]
    
    intervals = []
    
    # Process each line to create Interval objects
    for onset, offset, label in lines:
        # Create a new Interval object using the provided constructor or function called 'interval'
        intervals.append(
            interval(
                onset=int(onset),
                offset=int(offset),
                label=label
            )
        )
    # Return the list of Interval objects
    return intervals




def get_word_grouped_phones(words, phones, dictionary):
    """
    Groups phone intervals into lists based on word intervals and updates a dictionary with word-to-phone mappings.
    
    Args:
        words (list): A list of Interval objects representing word intervals.
        phones (list): A list of Interval objects representing phone intervals.
        dictionary (dict): A dictionary to store word-to-phone mappings.
        
    Returns:
        tuple: A tuple containing a list of lists of Interval objects representing word-grouped phone intervals, 
               and the updated dictionary.
    """
    word_grouped_phones = []
    word_idx = 0
    phones_grouped = []
    
    for ph in phones:
        if ph.label == 'SIL' and phones_grouped == []:
            word_grouped_phones.append([ph])
        # Add word grouped phones to list
        elif word_idx < len(words) and ph.offset == words[word_idx].offset:
            word = words[word_idx].label
            phones_grouped.append(ph)
            # Append phones_grouped to word_grouped_phones
            word_grouped_phones.append(phones_grouped)
            # Update the dictionary with the word-to-phone mapping
            dictionary[word] = " ".join([ph.label for ph in phones_grouped])
            # Reset phones_grouped and increment word_idx
            phones_grouped = []
            word_idx += 1
        else:
            # Append the phone interval to phones_grouped
            phones_grouped.append(ph)
    
    return word_grouped_phones, dictionary


def sannity_check(words, word_grouped_phones):
	"""
	This is just a sinity check to make sure that no overlaps have occured.
	"""
	words_grouped_nosil = [
		word for word in word_grouped_phones \
		if word[0].label != 'SIL'
	]
	assert len(words) == len(words_grouped_nosil)

def read_vowels(vowels_list):

	with open(vowels_list) as fd:
		return [ln.rstrip().upper() for ln in fd]


def get_audio(audio_path):
	return AudioSegment.from_wav(audio_path)

def adjust_time(word, time_adj):
    """
    Adjusts the onset and offset times of word intervals by a given time adjustment value.
    
    Args:
        word (list): A list of Interval objects representing word intervals.
        time_adj (int): The time adjustment value to be added to the onset and offset times.
        
    Returns:
        list: A list of Interval objects with adjusted onset and offset times.
    """
    # Create a new list to store the adjusted word intervals
    word_adj = [
        interval(
            onset=phone.onset + time_adj,
            offset=phone.offset + time_adj,
            label=phone.label
        ) for phone in word
    ]
    
    return word_adj


def get_num_word_to_repeat(idx, num_words):

	words_left = num_words - (idx + 1)  
	
	if words_left == 1 or words_left == 0:
		return 1
	elif words_left == 2:
		return 2
	else:
		return random.choice([2,3])

def append_multiple_words(words, corrupted_words, time_adj, words_dur):
    """
    Appends multiple words to the corrupted words list and updates the time adjustment variable.
    
    Args:
        words (list): A list of Interval objects representing words to be appended.
        corrupted_words (list): A list of objects interval_corrupted to store the corrupted word intervals.
        time_adj (int): The current time adjustment value.
        words_dur (int): The duration of the words being appended.
        
    Returns:
        tuple: A tuple containing the updated corrupted words list and the updated time adjustment value.
    """
    # Append adjusted word intervals to corrupted_words with label 3 
	# if more than one word (phrase repetition), otherwise label 2 (word repetition)
    for word in words:
        corrupted_words.append(
            interval_corrupted(
                word=adjust_time(word, time_adj),
                label=3 if len(words) > 1 else 2
            )
        )
    
    # Update time_adj by adding words_dur
    time_adj += words_dur
    
    # Append the original word intervals to corrupted_words with label 0
    for word in words:
        corrupted_words.append(
            interval_corrupted(
                word=adjust_time(word, time_adj),
                label=0
            )
        )
    
    # Return the updated corrupted_words list and the updated time_adj
    return corrupted_words, time_adj
	

def num_vowels_in_word(word, vowels):
	"""
	Get the number of vowels in a word.
	"""
	return len(
		[ph.label for ph in word if ph.label in vowels]
		)





def get_syllables(word_grouped):
	"""
    Returns the syllables in a word.
    
    Args:
        word_grouped (list): A list of Interval objects representing phone intervals.
        
    Returns:
        list: A list of lists representing the syllables in the word.
    """
	phones= [ph.label for ph in word_grouped]
	phones = [' '.join(phones)]
	syllables = generate(phones)
	syllables = [list(s.__str__()) for s in syllables]
	clean_syllables = [
		[ph.split()[0] for ph in syllable if ph != 'empty'] \
		for syllable in syllables
	]
	return clean_syllables

def get_syllables_to_repeat(word_grouped, syllables, num_syll_to_repeat):
    """
    Retrieves the phone intervals to repeat based on the given word_grouped, syllables, and number of syllables to repeat.
    
    Args:
        word_grouped (list): A list of Interval objects representing word-grouped phone intervals.
        syllables (list): A list of lists representing the syllables in the word.
        num_syll_to_repeat (int): The number of syllables to repeat.
        
    Returns:
        list: A list of phone intervals to repeat.
    """
    # Retrieve the syllables to repeat
    syllables_to_rep = syllables[0:num_syll_to_repeat]
    
    # Calculate the total number of phones to repeat
    num_phones_to_rep = sum([len(syll) for syll in syllables_to_rep])
    
    # Retrieve the corresponding phone intervals to repeat
    phones_to_rep = word_grouped[0:num_phones_to_rep]
    
    # Return the phone intervals to repeat
    return phones_to_rep

def corrupt(
	word_grouped_phones,
	vowels,
	corruption_rates = [0.1,0.2,0.3],
	disfluency_types = [1,2,3,4]
	):

	"""
	Applies corruption to word-grouped phone intervals based on given parameters.

	Args:
		word_grouped_phones (list): A list of Interval objects representing word-grouped phone intervals.
		vowels (list): A list of vowels.
		corruption_rates (list): A list of corruption rates to choose from.
		disfluency_types (list): A list of disfluency types.
		
	Returns:
		A tuple containing:
			the corrupted phone intervals grouped in words List[interval_corrupted], 
			audio segments info List[AudioSegmentsInfo],
			corruption types List[Int].

	Disfluency types:
		1: Part-word repetition
		2: Word repetition
		3: Phrase repetition
		4: Word Deletion

	"""

	num_words = len(word_grouped_phones)
	corruption_rate = random.choice(corruption_rates)
	num_corrupted_words = int(num_words*corruption_rate) + 1
	no_sil_indexes = [idx for idx, word in enumerate(word_grouped_phones) if word[0].label!='SIL']
	corrupt_word_idxes = random.sample(
		no_sil_indexes,
		num_corrupted_words
	)
	corruption_types = [
		random.sample(disfluency_types, 1)[0] for _ in range(num_corrupted_words)
	]
	
	corrupted = []
	time_adjust = 0 
	idx = 0 
	corr_idx = 0
	audio_segments_info = []
	while idx < len(word_grouped_phones):
		word_grouped = word_grouped_phones[idx]
		if idx in corrupt_word_idxes:
			corruption_type = corruption_types[corr_idx]
			# Part-Word Repetition
			num_vowels = num_vowels_in_word(word_grouped, vowels)
			if corruption_type and num_vowels > 0 == 1:
				# num_vowels = num_vowels_in_word(word_grouped, vowels)
				num_syllables_to_repeat = random.randint(1,num_vowels)
				syllables = get_syllables(word_grouped)
				syllables_to_rep = get_syllables_to_repeat(word_grouped, syllables, num_syllables_to_repeat)
				word_duration = syllables_to_rep[-1].offset - syllables_to_rep[0].onset
				syllables_to_repeat = interval_corrupted(
					word=adjust_time(syllables_to_rep, time_adjust),
					label=2 if num_syllables_to_repeat == 1 and num_vowels == 1 else 1
				)
				time_adjust+=word_duration
				word_orig = interval_corrupted(
					word=adjust_time(word_grouped, time_adjust),
					label=0
				)
				corrupted.append(syllables_to_repeat)
				corrupted.append(word_orig)
				audio_segments_info.append(
					AudioSegmentsInfo(
						onset=time_conv(syllables_to_rep[0].onset),
						offset=time_conv(syllables_to_rep[-1].offset),
						corr_type=1
					)
				)
				idx+=1
				corr_idx+=1
				pass
			# Word Repetition
			elif corruption_type == 2:
				word_duration = word_grouped[-1].offset - word_grouped[0].onset
				word_rep = interval_corrupted(
					word=adjust_time(word_grouped,time_adjust),
					label=2
				)
				time_adjust+=word_duration
				word_orig = interval_corrupted(
					word=adjust_time(word_grouped, time_adjust),
					label=0
				)
				corrupted.append(word_rep)
				corrupted.append(word_orig)
				audio_segments_info.append(
					AudioSegmentsInfo(
						onset=time_conv(word_grouped[0].onset),
						offset=time_conv(word_grouped[-1].offset),
						corr_type=1
					)
				)

				idx+=1
				corr_idx+=1
			# Phrase Repetition
			elif corruption_type == 3:
				num_rep_words = get_num_word_to_repeat(idx, num_words)
				words_duration = word_grouped_phones[idx+num_rep_words-1][-1].offset - word_grouped[0].onset
				phrase = word_grouped_phones[idx:idx+num_rep_words]
				corrupted, time_adjust = append_multiple_words(
					phrase,
					corrupted,
					time_adjust,
					words_duration
				)
				audio_segments_info.append(
					AudioSegmentsInfo(
						onset=time_conv(word_grouped[0].onset),
						offset=time_conv(word_grouped_phones[idx+num_rep_words-1][-1].offset),
						corr_type=1
					)
				)


				idx+=num_rep_words
				corr_idx+=1
			# Word Deletion
			else:
				word_duration = word_grouped[-1].offset - word_grouped[0].onset
				time_adjust-=word_duration
				audio_segments_info.append(
					AudioSegmentsInfo(
						onset=time_conv(word_grouped[0].onset),
						offset=time_conv(word_grouped[-1].offset),
						corr_type=0
					)
				)

				idx+=1
				corr_idx+=1
		else:
			corrupted.append(
				interval_corrupted(
					word=adjust_time(word_grouped, time_adjust),
					label=0
				)
			)			
			idx+=1 
	
	return corrupted, audio_segments_info, corruption_types



def corrupt_audio(audio, audio_segments_info):

	"""
    Applies corruption to the audio based on the provided audio segments info.
    
    Args:
        audio: The audio array.
        audio_segments_info List[AudioSegmentsInfo]: A list of AudioSegmentsInfo objects containing information
            about the corruption types and timings to repeat or delete.
            
    Returns:
        The corrupted audio array.
    """

	first_corruption_onset = audio_segments_info[0].onset
	audio_corrupted = audio[:first_corruption_onset]
	for idx, corruption_info in enumerate(audio_segments_info):
		onset = corruption_info.onset
		offset = corruption_info.offset
		next_onset = audio_segments_info[idx+1].onset if idx != len(audio_segments_info)-1 else -1
		# Repeat Segment
		if corruption_info.corr_type == 1:
			audio_corrupted = audio_corrupted.append(
				audio[onset: offset],
				crossfade=10
			)
			audio_corrupted = audio_corrupted.append(
				audio[onset: offset],
				crossfade=10
			)
			if offset == next_onset:
				continue
			else:
				audio_corrupted = audio_corrupted.append(
					audio[offset: next_onset],
					crossfade=10
				)
		# Delete Segment
		else:
			if offset == next_onset:
				continue
			else:
				audio_corrupted = audio_corrupted.append(
					audio[offset: next_onset],
					crossfade=10
				)

	return audio_corrupted


def write_corrupted_ground_truth(corrupted_words, path, path_words):
	phones = []
	words = []
	for word in corrupted_words:
		label = word.label
		word_list = []
		for phone in word.word:
			phones.append(
				[phone.label, phone.onset, phone.offset, label]
			)
			word_list.append(phone.label)
		words.append(word_list)
	with open(path, 'w') as f:
		writer = csv.writer(f)
		for phone in phones:
			writer.writerow(phone)

	with open(path_words, 'w') as f:
		for word in words:
			word = " ".join(word)
			f.write(f"{word}\n")



def write_corrupted_textGrid(corrupted_words, path):

	phones = []
	for word in corrupted_words:
		for phone in word.word:
			phones.append(
				(round(phone.onset/16000 ,4), round(phone.offset/16000, 4), phone.label)
			)
	tg = textgrid.Textgrid()
	phoneTier = textgrid.IntervalTier('phones', phones, 0, phones[-1][1])
	tg.addTier(phoneTier)
	tg.save(path,format="short_textgrid", includeBlankSpaces=False)


def sanity_timestampls(corrupted_words):
	phones = []
	for word in corrupted_words:
		for phone in word.word:
			phones.append(phone)
	# phones =  [item for sublist in phones for item in sublist]
	for idx, ph in enumerate(phones[:-1]):
		if ph.offset != phones[idx+1].onset:
			return False
	return True


def write_metadata(metadata, corrupted_dataset):
	path = os.path.join(corrupted_dataset, "metadata.csv")
	with open(path, 'w') as fd:
		writer = csv.writer(fd)
		writer.writerow(["name", "part_word", "word", "phrase", "deletion"])

		for data in metadata:
			name = data[0]
			corruption_types = data[1]
			part_word = corruption_types.count(1)
			word = corruption_types.count(2)
			phrase = corruption_types.count(3)
			deletion = corruption_types.count(4)
			writer.writerow(
				[name, part_word, word, phrase, deletion]
			)


def write_word_grouped_phones(word_grouped_phones, path):


	with open(path, 'w') as fd:
		for word in word_grouped_phones:
			phones = [ph.label for ph in word if ph.label != 'SIL']
			phones = " ".join(phones)
			fd.write(f"{phones}\n")



def time_conv(samples):

	return (samples/16000) * 1000

if  __name__ == '__main__':
	
	args = parse_arguments()
	phone_dir = args.phones_dir
	wav_dir = args.wavs_dir
	word_dir = args.words_dir
	output_dir = args.output_dir

	vowels = ['IY', 'IH', 'EH', 'EY', 'AE', 'AA', 'AW', 'AY', 'AH', 'AO', 'OY', 'OW', 'UH', 'UW', 'ER']



	phone_paths = [os.path.join(phone_dir, p) for p in os.listdir(phone_dir)]
	basenames = [os.path.basename(p)[:-3] for p in phone_paths]
	word_paths = [os.path.join(word_dir, p)+"wrd" for p in basenames]
	wav_paths = [os.path.join(wav_dir, p)+"wav" for p in basenames]


	audios_corrupted = os.path.join(output_dir, 'wavs')
	os.makedirs(audios_corrupted, exist_ok=True)
	phones_corrupted = os.path.join(output_dir, 'phones')
	os.makedirs(phones_corrupted, exist_ok=True)
	textgrid_corrupted = os.path.join(output_dir, 'textgrid')
	os.makedirs(textgrid_corrupted, exist_ok=True)
	word_grouped_phones_dir = os.path.join(output_dir, 'word_grouped_phones')
	os.makedirs(word_grouped_phones_dir, exist_ok=True)
	word_grouped_phones_gt_dir = os.path.join(output_dir, 'word_grouped_phones_gt')
	os.makedirs(word_grouped_phones_gt_dir, exist_ok=True)

	lexicon_path = os.path.join(output_dir, 'lexicon.txt')
	dictionary = {}
	metadata = []

	for phone_path, word_path, audio_path in zip(phone_paths, word_paths, wav_paths):

		audio_basename = os.path.basename(audio_path)
		phone_basename = os.path.basename(phone_path)
		basename = phone_basename[:-4]
		phone = get_intervals(phone_path)
		word = get_intervals(word_path)
		audio = AudioSegment.from_file(audio_path)
		audio_corrupted_path = os.path.join(audios_corrupted, audio_basename)
		phone_corrupted_path = os.path.join(phones_corrupted, phone_basename)
		textgrid_corrupted_path = os.path.join(textgrid_corrupted, phone_basename[:-3]+"textGrid")
		word_grouped_phones_path = os.path.join(word_grouped_phones_dir, phone_basename[:-3]+"WGP")
		word_grouped_phones_gt_path = os.path.join(word_grouped_phones_gt_dir, phone_basename[:-3]+"WGP")
		word_grouped_phones, dictionary = get_word_grouped_phones(word, phone, dictionary)
		corrupted, audio_segments_info, corruption_types = corrupt(word_grouped_phones, vowels)
		if not sanity_timestampls(corrupted): 
			raise  Exception("Overlap Error")
		else:
			metadata.append([basename, corruption_types])
			corr_audio = corrupt_audio(audio, audio_segments_info)
			corr_audio.export(audio_corrupted_path, format="wav")

			write_corrupted_textGrid(corrupted, textgrid_corrupted_path)
			write_corrupted_ground_truth(corrupted, phone_corrupted_path,	word_grouped_phones_gt_path)
			write_word_grouped_phones(word_grouped_phones, word_grouped_phones_path)
	with open(lexicon_path, 'w') as fd:
		for words, prons in dictionary.items():
			fd.write(f"{words} {prons}\n")
	write_metadata(metadata, output_dir)
