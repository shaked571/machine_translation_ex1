import itertools
import logging
import os
from collections import Counter
import numpy as np
from tqdm import tqdm
from typing import List
from multiprocessing import shared_memory, Process, Lock
from multiprocessing import cpu_count, current_process


"""
Download and extract the contents of this file.

You will find French data, English data, some gold alignments, an example of an alignment file output, some scripts, and a README file. 
Read the README file.

Goal: You need to write a program that takes the English and French data, aligns it, and produces an aligment file in the format described in the README file. In addition, you should write the translation parameters t(e,f) to a file.

For 80% of the credit, implment IBM model 1. For the additional 20%, implement model 2. You can find the details of model 2's EM procedure in Michael Collins' notes, available from the course webpage.

The supplied .a file contains manual alignments for the first few sentence pairs. This file should not be used in training your aligner (and it is too small anyways). But you can use it to measure the quality of your alignments in terms of AER using the supplied evaluation script.

What to submit: For this part, you need to submit a zip file containing (a) your code (b) a README file, describing how to run your code to align the data with model 1 and model 2 (c) output aligment file for model 1 and output aligment file for model 2.

A note about memory: While developing/debugging your code, you can use only the first k lines of the data. However, your final submission is expected to run on the entire dataset. This is a relatively small dataset, and should easily fit in the memory of a relatively modern personal computer. If it does not fit, maybe you should re-structure your program.
If you are using Java, make sure you allocate enough heap space for the JVM, as the default is very small. The head size can be specified using the -Xmx flag:

  java -Xmx1g YourClassName
will allocate 1gb of heap size.


"""


class Lang:
    base_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'hansards.')
    def __init__(self, suf):
        self.suf = suf
        self.data = self.read_file(self.base_name + self.suf)
        self.voc = Counter(self.flatten(self.data))
        self.unique = len(self.voc)
        self.w_index = {w: i for i, w in enumerate(self.voc)}

    @staticmethod
    def read_file(f_name: str) -> List[List[str]]:
        with open(f_name, encoding='utf-8') as f:
            lines = f.readlines()
        return [line.split() for line in lines]

    @staticmethod
    def flatten(t):
        return [item for sublist in t for item in sublist]

class IbmModel1:
    UNIQUE_NONE = '*None*'
    __lock = Lock()
    def __init__(self, source: Lang, target: Lang, n_ep=100, early_stop=True):
        self.logger = logging.getLogger("IBM_Model1")
        self.shm = shared_memory.SharedMemory(create=True, size=)

        self.source: Lang = source
        self.add_special_null()
        self.target: Lang = target
        self.n_ep: int = n_ep
        self.early_stop: bool = early_stop
        self.prob_ef_expected_alignment = self.init_uniform_prob()
        self.perplexities = [np.inf]
        self.algo()
        self.perplexities.pop(0)  # remove the first

    def add_special_null(self):
        self.source.voc[self.UNIQUE_NONE] = len(self.source.data)
        self.source.unique += 1
        self.source.w_index[self.UNIQUE_NONE] = self.source.unique - 1

    def calc_sent_pair(self,source_sent, target_sent, shm_countef, count_e_f, shm_total_f, total_f ):
        """
        # SENTENCE CONTEXT
        source_sent = [self.UNIQUE_NONE] + source_sent  # Adding Blank word in the beginning

        s_total = {w: 0 for w in source_sent}  # count
        for s_w, t_w in itertools.product(source_sent, target_sent):
            s_total[s_w] += self.expected_alignment(s_w, t_w)

        for s_w, t_w in itertools.product(source_sent, target_sent):
            expected = self.expected_alignment(s_w, t_w)
            collected_count = expected / s_total[s_w]
            count_e_f[self.source.w_index[s_w], self.target.w_index[t_w]] += collected_count
            total_f[self.target.w_index[t_w]] += collected_count
        """
        existing_shm = shared_memory.SharedMemory(self.shm.name)
        prob_ef = np.ndarray((self.source.unique, self.target.unique), dtype=np.float, buffer=existing_shm.buf)

        shm_countef = shared_memory.SharedMemory(shm_countef.name)
        count_e_f = np.ndarray(count_e_f.shape, dtype=np.float, buffer=shm_countef.buf)

        shm_total_f = shared_memory.SharedMemory(shm_total_f.name)
        total_f = np.ndarray((self.source.unique, self.target.unique), dtype=np.float, buffer=shm_total_f.buf)



        source_sent = [self.UNIQUE_NONE] + source_sent  # Adding Blank word in the beginning
        s_total = {w: 0 for w in source_sent}  # count

        for s_w, t_w in itertools.product(source_sent, target_sent):
            # expected = self.expected_alignment(s_w, t_w)
            expected = prob_ef[self.source.w_index[s_w], self.target.w_index[t_w]]
            collected_count = expected / s_total[s_w]
            self.__lock.acquire()
            count_e_f[self.source.w_index[s_w], self.target.w_index[t_w]] += collected_count
            total_f[self.target.w_index[t_w]] += collected_count
            self.__lock.release()

        existing_shm.close()
        shm_countef.close()
        shm_total_f.close()


    def create_shared_block(self, d1, d2=None):
        if d2 != None:
            a = np.zeros(shape=(d1, d2), dtype=np.int64)  # Start with an existing NumPy array
        else:
            a = np.zeros(shape=d1, dtype=np.int64)  # Start with an existing NumPy array

        shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
        # # Now create a NumPy array backed by shared memory
        np_array = np.ndarray(a.shape, dtype=np.int64, buffer=shm.buf)
        np_array[:] = a[:]  # Copy the original data into shared memory
        return shm, np_array


    def algo(self):

        for epoch in tqdm(range(self.n_ep), desc="epoch num", total=self.n_ep):
            curr_perp = self.calc_perp()
            if self.early_stop and curr_perp > self.perplexities[-1]:
                break
            self.logger.info(f"epoch {epoch} perplexity: {curr_perp}")
            self.perplexities.append(curr_perp)
            # E step
            shm_countef, count_e_f = self.create_shared_block(self.source.unique, self.target.unique)
            shm_total_f, total_f = self.create_shared_block(self.target.unique)  # all expected alignment of f (target)
            processes = []
            for i in range(cpu_count()):
                _process = Process(target=self.calc_sent_pair, args=(shr.name,))
                processes.append(_process)
                _process.start()

            for source_sent, target_sent in tqdm(zip(self.source.data, self.target.data),
                                                 desc="Iterate all sents pairs",
                                                 total=len(self.source.data)):


                calc_sent_pair()
                # SENTENCE CONTEXT
                source_sent = [self.UNIQUE_NONE] + source_sent  # Adding Blank word in the beginning

                s_total = {w: 0 for w in source_sent}  # count
                for s_w, t_w in itertools.product(source_sent, target_sent):
                    s_total[s_w] += self.expected_alignment(s_w, t_w)

                for s_w, t_w in itertools.product(source_sent, target_sent):
                    expected = self.expected_alignment(s_w, t_w)
                    collected_count = expected / s_total[s_w]
                    count_e_f[self.source.w_index[s_w], self.target.w_index[t_w]] += collected_count
                    total_f[self.target.w_index[t_w]] += collected_count

            # Update Prob
            # M step
            # F==target, e==source
            for s_w, t_w in itertools.product(self.source.voc, self.target.voc):
                upd_prob = count_e_f[self.source.w_index[s_w], self.target.w_index[t_w]] / total_f[self.target.w_index[t_w]]
                self.prob_ef_expected_alignment[self.source.w_index[s_w], self.target.w_index[t_w]] = upd_prob

    def expected_alignment(self, s_w, t_w):
        return self.prob_ef_expected_alignment[self.source.w_index[s_w], self.target.w_index[t_w]]

    def init_uniform_prob(self):
        # prepare shared memory
        a = np.ones((self.source.unique, self.target.unique))
        prob_ef = np.ndarray(a.shape, dtype=a.dtype, buffer=self.shm.buf)
        prob_ef[:] = a[:]
        prob_ef /= self.source.unique
        return prob_ef

    def calc_perp(self, ):
        # TODO
        pass
    """
    def perplexity(sentence_pairs, t, epsilon=1, debug_output=False):
    pp = 0
    
    for sp in sentence_pairs:
        prob = probability_e_f(sp[1], sp[0], t)
        pp += math.log(prob, 2) # log base 2
        
    pp = 2.0**(-pp)
    return pp

    """
"""
# Input: english sentence e, foreign sentence f, hash of translation probabilities t, epsilon 
# Output: probability of e given f

def probability_e_f(e, f, t, epsilon=1):
    l_e = len(e)
    l_f = len(f)
    p_e_f = 1
    
    for ew in e: # iterate over english words ew in english sentence e
        inner_sum = 0
        for fw in f: # iterate over foreign words fw in foreign sentence f
            inner_sum += t[(ew, fw)]
        p_e_f = inner_sum * p_e_f
    
    p_e_f = p_e_f * epsilon / (l_f**l_e)
    
    return p_e_f            

"""

if __name__ == '__main__':
    suf_fr = 'f'
    suf_en = 'e'
    suf_al = 'a'
    en = Lang(suf_en)
    fr = Lang(suf_fr)
    ibm1 = IbmModel1(en, fr, n_ep=4, early_stop=False)
