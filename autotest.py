import os

import generate_sentences

DEFAULT_SIZE = 500

INPUT_FILE = 'tests.txt'

PREFIX = 'res'

if __name__ == '__main__':
    # Generate sentences
    os.system('python main.py -g ' + str(DEFAULT_SIZE))

    # Read and Test
    with open("tests.txt", "r") as f:
        test_strs = f.readlines()
    
    for i in range(len(test_strs)):
        print("Testing No." + str(i + 1) + " sentence")
        # print('python main.py -s \"' + test_strs[i].rstrip('\n') + '\" > ' + str(i) + '.log')
        os.system('python main.py -s \"' + test_strs[i].rstrip('\n') + '\" > ' + str(i + 1) + '.log')
