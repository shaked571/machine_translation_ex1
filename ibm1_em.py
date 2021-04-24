import math
import sys
from datetime import datetime
from typing import List
import itertools
import logging
import os
from collections import Counter
import numpy as np
from tqdm import tqdm
import pickle
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
now = datetime.now().strftime("%d_%H_%M_%S")
if not os.path.isdir('logs'):
    os.mkdir('logs')
file_handler = logging.FileHandler(filename=os.path.join('logs', f'log_{now}'))
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)


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
        res = [line.split() for line in lines]#[:2000]
        print(len(res))
        return res

    @staticmethod
    def flatten(t):
        return [item for sublist in t for item in sublist]


class IbmModel1:
    UNIQUE_NONE = '*None*'
    saved_weight_fn = 'ibm1_p.npy'
    def __init__(self, source: Lang, target: Lang, n_ep=100, early_stop=True, init_from_saved_w=False, path_to_probs=None):
        self.logger = logging.getLogger("IBM_Model1")
        self.source: Lang = source
        self.add_special_null()
        self.target: Lang = target
        self.n_ep: int = n_ep
        self.early_stop: bool = early_stop
        if init_from_saved_w:
            self.prob_ef_expected_alignment = self.load_probs(path_to_probs)
        else:
            self.prob_ef_expected_alignment = self.init_uniform_prob()
            self.perplexities = [np.inf]
            self.algo()
            self.perplexities.pop(0)  # remove the first
            self.save_probs()

    def add_special_null(self):
        self.source.voc[self.UNIQUE_NONE] = len(self.source.data)
        self.source.unique += 1
        self.source.w_index[self.UNIQUE_NONE] = self.source.unique - 1

    def algo(self):

        for epoch in tqdm(range(self.n_ep), desc="epoch num", total=self.n_ep):
            curr_perp = self.calc_perp()
            self.logger.info(f"epoch {epoch} perplexity: {curr_perp}")
            self.perplexities.append(curr_perp)
            if self.early_stop and curr_perp - 20 > self.perplexities[-1]:
                break
            # E step
            count_e_f = np.zeros((self.source.unique, self.target.unique))
            total_f = np.zeros(self.target.unique)  # all expected alignment of f (target)
            for source_sent, target_sent in tqdm(zip(self.source.data, self.target.data),
                                                 desc="Iterate all sents pairs", total=len(self.source.data)):
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
            # M step
            for s_w in tqdm(self.source.voc, desc='calculating vocab', total=self.source.unique):
                for t_w in self.target.voc:
                    upd_prob = count_e_f[self.source.w_index[s_w], self.target.w_index[t_w]] / total_f[
                        self.target.w_index[t_w]]
                    self.prob_ef_expected_alignment[self.source.w_index[s_w], self.target.w_index[t_w]] = upd_prob

    def expected_alignment(self, s_w, t_w):
        return self.prob_ef_expected_alignment[self.source.w_index[s_w], self.target.w_index[t_w]]

    def init_uniform_prob(self):
        prob_ef = np.ones((self.source.unique, self.target.unique))
        prob_ef /= self.source.unique
        return prob_ef

    def calc_perp(self, ):
        prep = 0
        for source_sent, target_sent in tqdm(zip(self.source.data[:100], self.target.data[:100]),
                                             desc="Calc perp on  sents pairs",
                                             total=100):
            prob = self.probability_e_f(source_sent, target_sent)
            if prob != 0:
                prep += math.log(prob, 2)  # log base
        return -prep

    def probability_e_f(self, source_sent, target_sent, epsilon=1):
        source_len = len(source_sent)
        target_len = len(target_sent)
        p_e_f = 1
        for sw in source_sent:
            inner_sum = 0
            for tw in target_sent:
                inner_sum += self.expected_alignment(sw, tw)
            p_e_f = inner_sum * p_e_f

        p_e_f = p_e_f * epsilon / (target_len ** source_len)

        return p_e_f

    def predict(self, source_sent, target_sent):
        res = []
        for s_idx, sw in enumerate(source_sent):
            curr_p = self.expected_alignment(sw, self.UNIQUE_NONE)
            probable_align = None

            for t_idx, tw in enumerate(target_sent):
                align_prob = self.expected_alignment(sw, tw)
                if align_prob >= curr_p:  # prefer newer word in case of tie
                    curr_p = align_prob
                    probable_align = t_idx

            res.append(f"{s_idx}-{probable_align}")

        return " ".join(res)

    def save_probs(self):
        with open(self.saved_weight_fn, 'wb') as f:
            np.save(f, self.prob_ef_expected_alignment)


    def load_probs(self, path_to_probs):
        if path_to_probs is None:
            path = self.saved_weight_fn
        else:
            path = os.path.join(path_to_probs, self.saved_weight_fn)
        with open(path, 'rb') as f:
            prob_ef_expected_alignment = np.load(f)
        return prob_ef_expected_alignment


if __name__ == '__main__':
    suf_fr = 'f'
    suf_en = 'e'
    suf_al = 'a'
    en = Lang(suf_en)
    fr = Lang(suf_fr)
    ibm1 = IbmModel1(en, fr, n_ep=50, init_from_saved_w=False, early_stop=True)

