import os
import sys
from copy import copy
from itertools import permutations

import numpy as np
import enchant
from collections import Counter, deque

import pandas as pd
from tqdm import tqdm
import hashlib


class AnagramSolver(object):
    """ Class that provides methods for generating anagrams and checking them against an MD5 hash """

    def __init__(self, wordsfile, anagram, hashes, only_real_words=False):
        """ initialise anagram solver

        :param wordsfile: file containing list of words. expected to have one word per line
        :param anagram: a string representing any anagram
        :param hashes: a list of hash keys to check generated anagrams against
        """
        # anagram to compare against
        self.word_counts = {}
        self.word_sets = {}
        self.word_counters = {}
        self.anagram = anagram
        self.allowable_counts = Counter(anagram)
        self.anagram_np = pd.Series(Counter(anagram), dtype=int).reindex(sorted(list(set(anagram))), axis=1).values

        self.hashes = hashes
        # create a numpy vector of words (numpy is faster to process)
        self.words = np.genfromtxt(wordsfile, dtype=str)
        print(f"number of words in initial list = {self.words.shape[0]}")

        # filter out duplicates
        self.words = np.unique(self.words)
        print(f"number of words after removing duplicates = {self.words.shape[0]}")

        # filter out all the words that
        self.filter_base_words(only_real_words)

        pd_list = []
        for idx, word in enumerate(self.words):
            pd_list.append(pd.DataFrame(Counter(word), index=[idx], dtype=int))
        words_pd = pd.concat(pd_list)
        # get a numpy array with column for each letter e.g. where value is count of that letter. e.g
        #        A  B  C  D
        # A:     1  0  0  0
        # AB:    1  1  0  0
        # AC:    1  0  1  0
        self.words_np = words_pd.fillna(0).reindex(sorted(words_pd.columns), axis=1).values

    def filter_base_words(self, only_real_words=False):
        """ function to remove the words in the words list that are possible with this anagram"""


        self.set_details()
        # remove words that have unallowed characters
        rm_bad_words = np.vectorize(self._contains_allowed_chars)
        self.words = self.words[rm_bad_words(self.words)]
        print(f"number of words after removing unallowed characters = {self.words.shape[0]}")

        if only_real_words:
            # filter out words that are not real words
            self.words = self._keep_real_words(self.words)
            print(f"number of words after removing words that aren't in dictionary = {self.words.shape[0]}")

        # replace apostrophes in words with nothing. anagram does not have apostrophes, and duplicates might be formed
        # once apostrophes have been removed e.g. "harrys" and "harry's"
        rm_apostrophe = np.vectorize(self._remove_apostrophe)
        words_no_apostrophy = rm_apostrophe(self.words)
        self.words = np.unique(words_no_apostrophy)
        print(f"number of words (after removal of apostrophies and duplicates) = {self.words.shape[0]}")

        # remove words that are not compatible with other words
        #rm_uncompatible_words = np.vectorize(self.compatible_with_other_words)
        #self.words = self.words[rm_uncompatible_words(self.words)]
        #print(f"number of words after removing words incompatible with all other words = {self.words.shape[0]}")

        self.set_details()

    def set_details(self):
        """ sets details for the words list, so that they don't need to be re-calculated

        :return: Nothing, but updates calcs with self.word_counts, self.word_sets and self.word_counters
        """
        self.word_counts = {}
        self.word_sets = {}
        self.word_counters = {}
        for word in self.words:
            self.word_counts[word] = len(word)
            self.word_sets[word] = set(word)
            self.word_counters[word] = Counter(word)

    def generate_anagrams(self, no_words):
        """ generate all anagrams of string provided"""
        # warning this is a recursive function and may take a long time to process
        self._generate_anagrams(self.words_np, self.anagram_np, np.array(range(self.words_np.shape[0])), no_words)


    def _word_combo_to_anagrams(self, word_list):
        """ convert a word combination to a list of anagrams, and check against hashes

        :param word_list: a list of of words e.g ["dog","cat"]
        :return: nothing, but prints if correct anagram is found
        """

        # generate all permutations of word combo
        perms = permutations(word_list)
        # for each permutation, create md5 hash string, compare against provided hashes and print if there is a match
        for perm in perms:
            anagram = " ".join(perm).encode('utf-8')
            hash_str = hashlib.md5(anagram).hexdigest()
            if hash_str in self.hashes:
                idx = self.hashes.index(hash_str)
                print(f"Correct match found for hash {self.hashes[idx]}!: {anagram}")

    def _generate_anagrams(self, words_np, anagram_np, indices, remaining_word_count, firstlevel=True):
        """ Generates all the anagrams of a specified number of words and checks them against hashes

        Note this function is recursive. When called (firstlevel=true), tqdm is used to provide progress update. the
        function recurses for each additional word that it has to find to make the anagram. e.g. The function is called,
        finds a word, then calls itself where it finds a second word. If only 3 words are allowed, in the same function
        call the function then finds if any words match the remaining letters to produce a word combination that can
        create anagrams.

        :param allowed_words: a numpy array of allowed words
        :param char_counter: a collections.counter object of the anagram
        :param remaining_word_count: The number of words to process to make an anagram with
        :param firstlevel: whether this is the highest level of the function call
        :return: Nothing

        """
        # create iterator with tqdm if first level
        if firstlevel:
            iterator = tqdm(zip(indices, words_np), desc="number words assessed for all combinations",
                            total=words_np.shape[0])
        else:
            iterator = zip(indices, words_np)

        all_word_combos = deque() # quicker than using lists

        for (j, (i, word)) in enumerate(iterator):

            # Get rid of 'word' from arrays. we will process ever combination of this word and so it does not need
            # to be checked again
            words_np = words_np[1:, :]
            indices = indices[1:]
            # get updated numpy arrays with current word removed
            new_anagram_np, remaining_words_np, remaining_indices = self._remove_word(anagram_np, word, words_np, indices)

            # we have found an anagram in less words than the max amount of words specified
            if new_anagram_np.sum() == 0:
                all_word_combos.extend([[i]])
                continue

            # there are no anagrams the words currently selected, as there are no remaining words compatible
            if remaining_words_np.shape[0] == 0:
                continue
            # if we only need to get one more word, do this now rather than recursively again
            if (remaining_word_count - 1) == 1:
                # get all words where the total number of characters matches the anagram. These words will work as we
                # have already checked that they contain the correct letters
                mask = remaining_words_np.sum(axis=1) == new_anagram_np.sum()
                # add all the remaining word integers to the word combo deque
                remaining_indices = remaining_indices[mask]
                for i2 in range(remaining_indices.shape[0]):
                    all_word_combos.append([i, remaining_indices[i2]])
                continue

            # if we have gotten here then we need to find at least 2 more words. do this recursively
            word_combos = self._generate_anagrams(remaining_words_np, new_anagram_np, remaining_indices,
                                                  remaining_word_count - 1, False)
            # add all the word combinations indices found
            for other_words in word_combos:
                all_word_combos.append(other_words + [i])
            # if we have made it back up to the first level, then we are ready to check the word combinations found
            # against the hashes provided. remove the word cominations from deque so that it doesnt grow too large.
            if firstlevel:
                while len(all_word_combos) > 0:
                    val = list(map(lambda i: self.words[i], all_word_combos.pop()))
                    self._word_combo_to_anagrams(val)

        return all_word_combos

    def _remove_word(self, anagram_np, word, words_np, indices):
        """ given a word, it removes any words that can not form an anagram of anagram_pd with this word

        :param anagram_np: numpy array of anagram, where columns represent each character and values represent count
        :param word: numpy array of a word, where columns represent each character and values represent count
        :param words_np:    numpy array of words list, where columns represent each character, rows for each word and
                            values represent count
        :param indices:     numpy array of indices in global words lists associated with words_pd
        :return:
        """
        # this is the remaining character counts after removing word from anagram_pd
        new_anagram_pd = anagram_np - word

        # where each word has less characters than the remaining anagram characters
        mask = words_np <= new_anagram_pd
        # if all values in row = True, then word is allowed as word can be contained within anagram.
        # min of [true, true] is true
        # if any value in row = False, then word has too many characters of one type. min of [true, false] is false
        mask2 = mask.min(axis=1)
        # filter down to list of remaining allowable words and associated indices of these words
        remaining_words_pd = words_np[mask2]
        indices = indices[mask2]
        return new_anagram_pd, remaining_words_pd, indices

    def compatible_with_other_words(self, word):
        """ checks a word is not compatible with any other words.

        :param word: str, word to check
        :param return_words:
        :return: Bool, True is word is compatible with others, False if not
        """
        word_counter = Counter(word)
        original_counter = copy(self.allowable_counts)
        reduced_counter = copy(self.allowable_counts)
        for k in word_counter:
            reduced_counter[k] -= word_counter[k]

        # temporarily overwrite counter
        self.allowable_counts = reduced_counter

        # remove words that have un-allowed characters
        rm_bad_words = np.vectorize(self._contains_allowed_chars)
        allowed_words = self.words[rm_bad_words(self.words)]

        # reset char counter
        self.allowable_counts = original_counter

        if allowed_words.shape[0] == 0:
            return False
        else:
            return True


    @staticmethod
    def _keep_real_words(words, dict_type="en_GB"):
        """ keep only words found in dictionary

        :param dict_type: type of dictionary, must be a dictionary type used by pyenchant, default is "en_GB"
        :return: None, but updates self.words
        """
        d = enchant.Dict(dict_type)
        check_word = np.vectorize(d.check)
        real_words_mask = check_word(words)
        return words[real_words_mask]


    @staticmethod
    def _remove_apostrophe(word):
        """ Remove apostrophe from word """
        return word.replace("'","")

    def _contains_allowed_chars(self, word):
        """ check if word contains allowed characters"""
        word_set = self.word_sets[word]
        counter = self.word_counters[word]

        allowable_chars = set(self.allowable_counts.keys())

        if word_set.intersection(allowable_chars) == word_set:
            for char in counter:
                if counter[char] > self.allowable_counts[char]:
                    return False
            return True
        else:
            return False


if __name__ == "__main__":

    try:
        words_file, hashes_file, anagram, max_words = sys.argv[-4:]
    except ValueError as e:
        raise ValueError("Please run this program as follows: python anagram_solver.py <word_list.txt> <hashes.tx> "
                         "<anagram> <max number of words>")

    for txt, path in zip(["<words_file>", "<hashes_file>"], [words_file, hashes_file]):
        if not os.path.exists(words_file):
            raise ValueError(f"{txt} parameter does not point to valid path")

    try:
        max_words = int(max_words)
    except ValueError as e:
        raise ValueError("<max number of words> must be an integer")

    with open(hashes_file, 'r') as f:
        hashes = [hsh.strip("\n") for hsh in f.readlines()]

    anagram_solver = AnagramSolver(words_file, anagram, hashes)

    # this will find all anagrams up to 4 words long
    anagram_solver.generate_anagrams(max_words)

