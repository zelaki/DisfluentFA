import k2
import numpy as np






class FSA:
    def __init__(self):
        self.arcs = []

    def add_arc(self, start_state, end_state, input_symbol, weight):
        arc = f"{start_state} {end_state} {input_symbol} {weight}"
        self.arcs.append(arc)

    def get_fst_string(self):
        last_start_state = self.arcs[-1].split()[0]
        final_start_state = int(last_start_state) + 1
        final_end_state = final_start_state + 1
        arc = f"{final_start_state} {final_end_state} -1 0"
        self.arcs.append(arc)
        self.arcs.append(f'{final_end_state}')
        fsa_str = '\n'.join(self.arcs)
        fsa = k2.Fsa.from_str(fsa_str)
        sorted_fsa = k2.arc_sort(fsa)
        return sorted_fsa


def get_modified_fsa(tokens, start_word_indexes, T=0.):
    start_word_indexes = [idx+1 for idx in start_word_indexes]
    end_word_indexes = [idx-1 for idx in start_word_indexes]
    start_word_indexes = [1] + start_word_indexes
    word_counter = 0
    fsa=FSA()
    fsa.add_arc(0, start_word_indexes[1], 0, np.log((1-T)/2))


    for idx, token in enumerate(tokens):

        # We are at the ending of a word
        if idx-1 in end_word_indexes:
            fsa.add_arc(idx, idx+1, token,  np.log(T))

            # Add ARC to previous word (model word repetition) 
            fsa.add_arc(idx, start_word_indexes[word_counter], 0, np.log((1-T)/2))

            # Add ARC to next word (model word deletion) 
            try:
                fsa.add_arc(idx, start_word_indexes[word_counter+2], 0, np.log((1-T)/2))
            except IndexError:
                pass
            word_counter+=1
        else: 
            # Add ARC to begining of current word (modelpart word repetitions)
            if idx not in start_word_indexes: 
                fsa.add_arc(idx, start_word_indexes[word_counter], 0, np.log(1-T))
                        
            fsa.add_arc(idx, idx+1, token, round(np.log(T)))

    sorted_fsa = fsa.get_fst_string()

    return sorted_fsa





