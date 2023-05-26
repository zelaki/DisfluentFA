import sys
import os


def read_phones_to_list(path):
    with open(path) as fd:
        lines = [ln.rstrip().split(',') for ln in fd]
    return lines


def write_phone_list(phone_list, output_path):
    with open(output_path, 'w') as fd:
        for phone in phone_list:
            fd.write(f'{phone[0]},{phone[1]},{phone[2]},{phone[3]}\n')

def fix_dub(phone_list):
    """
    list format: [[onset, offset, phone], [...]]
    """
    idx = 0 
    phone_list_fixed = []
    while idx < len(phone_list):
        try:
            start_time = phone_list[idx][1]
            phone = phone_list[idx][0]
            label = phone_list[idx][3]
            next_end_time = phone_list[idx+1][2]
            next_phone = phone_list[idx+1][0]


        except: IndexError
        
        if phone == next_phone:
            phone_list_fixed.append([phone, start_time, next_end_time, label])
            idx+=2
        else:
            phone_list_fixed.append(phone_list[idx])
            idx+=1
    return phone_list_fixed


if __name__ == '__main__':
        
    phone_dir = sys.argv[1]
    output_dir = sys.argv[2]
    phone_paths = [os.path.join(phone_dir, p) for p in os.listdir(phone_dir)]
    output_paths = [os.path.join(output_dir, os.path.basename(p)) for p in phone_paths]


    for phone_path, output_path in zip(phone_paths, output_paths):
        phone_list = read_phones_to_list(phone_path)
        fixed_phone_path = fix_dub(phone_list)
        write_phone_list(fixed_phone_path, output_path)
