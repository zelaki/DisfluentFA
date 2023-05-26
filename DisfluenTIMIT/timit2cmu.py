import sys
import os
from dataclasses import dataclass

@dataclass
class interval():
	onset: int
	offset: int
	phone: str


def get_timit2cmu(path):
	with open(path) as fd:
		lines = [ln.rstrip().split() for ln in fd]
	timit2cmu = {}
	for ln in lines:
		timit2cmu[ln[0]] = ln[1]
	return timit2cmu
 


def timit2cmu_covert(phones_path, timit2cmu_map):
	cmu_phones = []
	with open(phones_path) as fd:
		lines = [ln.rstrip().split() for ln in fd]
	for onset, offset, phone in lines:
		cmu_phones.append(interval(
			onset=onset,
			offset=offset,
			phone=timit2cmu_map[phone].upper()
		))
	return cmu_phones

def write_phones(phones, out_path):

	with open(out_path, "w") as fd:
		for inter in phones:
			fd.write(
				"{} {} {}\n".format(
					inter.onset,
					inter.offset,
					inter.phone
				)
			)


timit_phones_dir = sys.argv[1]
timit2cmu_map = sys.argv[2]
cmu_phones_dir = sys.argv[3]

timit_phones_paths = [os.path.join(timit_phones_dir, p) for p in os.listdir(timit_phones_dir)]
timit2cmu_map = get_timit2cmu(timit2cmu_map)

for path in timit_phones_paths:

	cmu_phones = timit2cmu_covert(path, timit2cmu_map)
	bname = os.path.basename(path)
	out_path = os.path.join(cmu_phones_dir, bname)
	write_phones(cmu_phones, out_path)
