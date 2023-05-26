
mkdir -p timit
mkdir -p timit/phones
mkdir -p timit/phones_cmu
mkdir -p timit/words
mkdir -p timit/wavs

word_dir=timit/words
phone_dir=timit/phones
phone_cmu_dir=timit/phones_cmu
wav_dir=timit/wavs


# Change the folder structure
for f in $(find $1 -print | grep wrd); do
    # delete base dir from path
    removed_base_dir_path="${f#*\/}"
    #replace / with _
    dest_path="${removed_base_dir_path//\//_}"
    cp $f $word_dir/$dest_path
done

for f in $(find $1 -print | grep phn); do
    # delete base dir from path
    removed_base_dir_path="${f#*\/}"
    #replace / with _
    dest_path="${removed_base_dir_path//\//_}"
    cp $f $phone_dir/$dest_path
done

for f in $(find $1 -print | grep wav); do
    # delete base dir from path
    removed_base_dir_path="${f#*\/}"
    #replace / with _
    dest_path="${removed_base_dir_path//\//_}"
    cp $f $wav_dir/$dest_path
done

# Covert TIMIT to CMU phones
python3 timit2cmu.py $phone_dir timit2cmu_phones.map $phone_cmu_dir
python3 corrupt_timit.py --phones_dir  $phone_cmu_dir \
                         --words_dir $word_dir \
                         --wavs_dir $wav_dir \
                         --output_dir  disfluent_timit
rm -rf timit

# Since we map the closure symbols bcl,dcl,gcl,pcl,tck,kcl for the stops b,d,g,p,t,k 
# in the same CMU phones, duplicates are created. Here we merge dublicate phones.
mkdir phones_fixed
python3 fix_dublicates.py disfluent_timit/phones/ phones_fixed
rm -rf disfluent_timit/phones/ 
mv phones_fixed disfluent_timit/phones/