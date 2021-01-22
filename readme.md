# AnagramSolver Readme

This python package provides a CLI programme to generate anagrams given a words list, a string, and a max number of 
words, and then compares the generated anagrams against a list of user provided hashes

## Setup
This python package uses python 3.7. the file `anagramcracker.yml`  as been provided which can be used to build the 
conda environment required to run anagramsolver.py

## Running
To run this package, please enter the following command:

`python anagramsolver.py <word_list.txt> <hashes.txt> <anagram> <max number of words>` 

where:
- `word_list.txt` is a text file containing the words to use in the anagram generation. Note that this should only 
include one word per line. A file called 'wordlist.txt' has been provided as part of this tool
- `hashes.txt` is a text file containing the MD5 hashes to assess the anagrams against. Note that this should only 
include one hash per line. An example called 'hashes.txt' has been provided which provides the following hashes for the 
anagram 'poultryoutwitsants':
    - 23170acc097c24edb98fc5488ab033fe: 'ty outlaws printouts'
    - e4820b45d2277f3844eac66c903e84be: 'printout stout yawls'
    - 665e5bcb0c20062fe8abaaf4628bb154: 'wu lisp not statutory'
- `anagram` is a string that you would like to generate anagrams for
- `max number of words` is an integer defining the maximum number of words that an anagram can consist of. as the number 
of words goes up, the number of word combinations and thus anagrams increases significantly. all anagrams with less words 
than the value specified with be generated as well. 

##Testing
A test script has been provided which validates the word combination generation part of the solver. the test script can be ran 
with py test using the following command:

`python -m pytest -v`


