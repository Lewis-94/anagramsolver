import numpy as np

from anagramsolver import AnagramSolver

def test_word_combo_generation():
    """ test word combination generation against a known set of anagrams for the string 'dogcat' """
    words_list = "wordlist.txt"
    no_words = 2

    # intiialise anagramsolver with the string 'dogcat' and existing set of words
    ag_solver = AnagramSolver(words_list, "dogcat", [""], True)

    # generate the indices of the word combinations that form anagrams
    word_combos_idxs = ag_solver._generate_anagrams(ag_solver.words_np, ag_solver.anagram_np,
                                               np.array(range(ag_solver.words_np.shape[0])), 2)

    # convert the indices into the actual words, and turn this into a set, to make sure order doesn't matter
    word_combos = []
    for i, i2 in word_combos_idxs:
        word_combos.append((ag_solver.words[i], ag_solver.words[i2]))
    word_combos = set(word_combos)

    # this has been generated independently online
    actual_word_combos = {('act', 'dog'), ('act', 'god'), ('cad', 'got'), ('cad', 'tog'), ('cd', 'goat'),
                          ('cd', 'toga'), ('cad', 'got'), ('cad', 'tog'), ('cat', 'dog'), ('cat', 'god'),
                          ('cod', 'tag'), ('cog', 'tad'), ('cot', 'dag'), ('cot', 'gad'), ('ct', 'dago'),
                          ('ct', 'goad'), ('dag', 'oct'), ('dc', 'goat'), ('dc', 'toga'), ('doc', 'tag'),
                          ('gad', 'oct')}

    assert word_combos == actual_word_combos

if __name__ == "__main__":
    
    test_word_combo_generation()