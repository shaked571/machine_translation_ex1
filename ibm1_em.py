import abc
import argparse
import functools
import math
import sys
from datetime import datetime
from typing import List
import itertools
import logging
import os
from collections import Counter, defaultdict
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

    def __init__(self, suf, number_of_lines=None):
        self.suf = suf
        self.logger = logging.getLogger(f"lang-{self.suf}")
        self.number_of_lines = number_of_lines
        self.data = self.read_file(self.base_name + self.suf)
        self.voc = Counter(self.flatten(self.data))
        self.unique = len(self.voc)
        self.w_index = {w: i for i, w in enumerate(self.voc)}

    def read_file(self, f_name: str) -> List[List[str]]:
        with open(f_name, encoding='utf-8') as f:
            lines = f.readlines()

        res = [line.split() for line in lines]
        if self.number_of_lines != None:
            res = res[:self.number_of_lines]
        self.logger.info(f"using {len(res)} lines")
        return res

    @staticmethod
    def flatten(t):
        return [item for sublist in t for item in sublist]


class IbmModel(abc.ABC):
    UNIQUE_NONE = '*None*'

    def __init__(self, source: Lang, target: Lang, n_ep=100,early_stop=True ):
        self.model_name = None
        self.target: Lang = target
        self.source: Lang = source
        self.n_ep: int = n_ep
        self.early_stop: bool = early_stop
        self.prob_ef_expected_alignment = None


    def expected_alignment(self, s_w, t_w):
        return self.prob_ef_expected_alignment[t_w][s_w]


    def add_special_null(self, lang):
        lang.voc[self.UNIQUE_NONE] = len(lang.data)
        lang.unique += 1
        lang.w_index[self.UNIQUE_NONE] = lang.unique - 1
        return lang

    def calc_perp(self, ):
        prep = 0
        for source_sent, target_sent in tqdm(zip(self.source.data[:100], self.target.data[:100]),
                                             desc="Calc perp on  sents pairs",
                                             total=100):
            prob = self.probability_e_f(source_sent, target_sent)
            if prob != 0:
                prep += math.log(prob, 2)  # log base
        return -prep

    def init_uniform_prob(self):
        prob_ef = defaultdict(lambda: defaultdict(lambda: 1 / self.target.unique))
        return prob_ef

    def predict_all(self):
        res = []
        for source_sent, target_sent in tqdm(zip(self.source.data, self.target.data),
                                             desc="preddicting sents", total=len(self.source.data)):
            res.append(self.predict(source_sent, target_sent))
        with open(f"prediction_{self.model_name}.txt", mode='w') as f:
            f.writelines(res)

    @abc.abstractmethod
    def predict(self, source_sent, target_sent):
        pass

    @abc.abstractmethod
    def probability_e_f(self, source_sent, target_sent):
        pass


class IbmModel1(IbmModel):
    saved_weight_fn = 'ibm1_p.pkl'
    def __init__(self, source: Lang, target: Lang, n_ep=100, early_stop=True, init_from_saved_w=False, path_to_probs=None, saved_weight_fn=None):
        super(IbmModel1, self).__init__(source, target, n_ep, early_stop )
        self.model_name = 'IBM_Model1'
        self.logger = logging.getLogger("IBM_Model1")
        self.source = self.add_special_null(self.source)
        if init_from_saved_w:
            self.logger.info("loading...")
            self.prob_ef_expected_alignment = self.load_probs(path_to_probs)
        else:
            self.prob_ef_expected_alignment = self.init_uniform_prob()
            self.perplexities = [np.inf]
            self.algo()
            self.perplexities.pop(0)  # remove the first
            self.save_probs()

    def load_probs(self, path_to_probs):
        if path_to_probs is None:
            path = self.saved_weight_fn
        else:
            path = os.path.join(path_to_probs, self.saved_weight_fn)
        with open(path, 'rb') as f:
            prob = pickle.load(f)

        return prob


    def algo(self):
        for epoch in tqdm(range(self.n_ep), desc="epoch num", total=self.n_ep):
            curr_perp = self.calc_perp()
            self.logger.info(f"epoch {epoch} perplexity: {curr_perp}")
            self.perplexities.append(curr_perp)
            print(curr_perp)
            print(self.perplexities[-2])
            if self.early_stop and curr_perp + 1 > self.perplexities[-2]:
                self.logger.info("Doing early stopping, the model converged.")
                break
            # E step
            count_e_f = defaultdict(lambda: defaultdict(int))
            total_f = defaultdict(int)  # all expected alignment of f (target)
            for source_sent, target_sent in tqdm(zip(self.source.data, self.target.data),
                                                 desc="Iterate all sents pairs", total=len(self.source.data)):
                # SENTENCE CONTEXT
                source_sent = [self.UNIQUE_NONE] + source_sent  # Adding Blank word in the beginning

                s_total = defaultdict(int)  # count
                for s_w in source_sent:
                    for t_w in target_sent:
                        s_total[s_w] += self.expected_alignment(s_w, t_w)

                for s_w in source_sent:
                    for t_w in target_sent:
                        expected = self.expected_alignment(s_w, t_w)
                        collected_count = expected / s_total[s_w]
                        count_e_f[t_w][s_w] += collected_count
                        total_f[t_w] += collected_count
            # M step
            for t_w, t_w_count in tqdm(count_e_f.items(), desc='calculating vocab', total=len(count_e_f)):
                for s_w, val in t_w_count.items():
                    upd_prob = val / total_f[t_w]
                    self.prob_ef_expected_alignment[t_w][s_w] = upd_prob





    def probability_e_f(self, source_sent, target_sent):
        source_len = len(source_sent)
        target_len = len(target_sent)
        p_e_f = 1
        for sw in source_sent:
            inner_sum = 0
            for tw in target_sent:
                inner_sum += self.expected_alignment(sw, tw)
            p_e_f = inner_sum * p_e_f

        p_e_f = p_e_f / (target_len ** source_len)

        return p_e_f

    def predict(self, source_sent, target_sent):
        res = []
        for t_idx, tw in enumerate(target_sent):
            best_prob = self.expected_alignment(self.UNIQUE_NONE, tw)
            probable_align = self.UNIQUE_NONE
            for s_idx, sw in enumerate(source_sent):
                curr_prob = self.expected_alignment(sw, tw)
                if curr_prob >= best_prob:
                    best_prob = curr_prob
                    probable_align = s_idx
            if probable_align != self.UNIQUE_NONE:
                res.append(f"{t_idx}-{probable_align}")
        str_out = " ".join(res)
        str_out =str_out + "\n"
        return str_out

    def save_probs(self):
        self.logger.info("Saving probs...")
        for k, v in self.prob_ef_expected_alignment.items():
            self.prob_ef_expected_alignment[k] = dict(v)
        self.prob_ef_expected_alignment = dict(self.prob_ef_expected_alignment)

        with open(self.saved_weight_fn, 'wb') as f:
            pickle.dump(dict(self.prob_ef_expected_alignment), f, pickle.HIGHEST_PROTOCOL)





#1. Add lidstone smoothing
#2. Adding null to the source sentence
#My ideas:
#2. Do a normalization over the vocabulary - 'lower' to make unique.
# Add extra null words?

class IbmModel2(IbmModel):
    saved_weight_fn = 'ibm2_p.pkl'
    saved_distortion_fn = 'ibm2_distortion.pkl'
    DUMMY_WORD  = "*DUMMY*FOR*LEN*"
    def __init__(self, source: Lang, target: Lang, n_ep=100, early_stop=True, init_from_saved_w=False,
                 path_to_probs=None, saved_weight_fn=None, path_to_distortion=None):
        super().__init__(source, target, n_ep, early_stop)
        self.model_name = "IBM_Model2"
        self.logger = logging.getLogger(self.model_name)
        self.source: Lang = self.add_special_null(source)
        #Need to add length of both sentences
        self.n_ep: int = n_ep
        if saved_weight_fn is not None:
            self.saved_weight_fn = saved_weight_fn

        if init_from_saved_w:
            self.logger.info("loading...")
            self.prob_ef_expected_alignment = self.load_probs(path_to_probs, self.saved_weight_fn)
            self.distortion_table = self.load_probs(path_to_distortion, self.saved_distortion_fn)

        else:
            self.prob_ef_expected_alignment = self.init_uniform_prob()
            self.distortion_table = self.init_distortion_uniformly()
            self.perplexities = [np.inf]
            self.algo()
            self.perplexities.pop(0)  # remove the first
            self.save_probs()

    def load_probs(self, path_to_probs, file_name):
        if path_to_probs is None:
            path = file_name
        else:
            path = os.path.join(path_to_probs, file_name)
        with open(path, 'rb') as f:
            prob = pickle.load(f)

        return prob

    def save_probs(self):
        self.logger.info("Saving probs...")
        for k, v in self.prob_ef_expected_alignment.items():
            self.prob_ef_expected_alignment[k] = dict(v)
        self.prob_ef_expected_alignment = dict(self.prob_ef_expected_alignment)
        with open(self.saved_weight_fn, 'wb') as f:
            pickle.dump(dict(self.prob_ef_expected_alignment), f, pickle.HIGHEST_PROTOCOL)

        for k, v in self.distortion_table.items():
            self.prob_ef_expected_alignment[k] = dict(v)
            for k1, v1 in v.items():
                v[k1] = dict(v1)
                for k2, v2 in v1.items():
                    v1[k2] = dict(v2)
                    for k3, v3 in v2.items():
                        v2[k3] = dict(v3)
        self.prob_ef_expected_alignment = dict(self.prob_ef_expected_alignment)
        with open(self.saved_weight_fn, 'wb') as f:
            pickle.dump(dict(self.prob_ef_expected_alignment), f, pickle.HIGHEST_PROTOCOL)



    def algo(self):
        for epoch in tqdm(range(self.n_ep), desc="epoch num", total=self.n_ep):
            curr_perp = self.calc_perp()
            self.logger.info(f"epoch {epoch} perplexity: {curr_perp}")
            self.perplexities.append(curr_perp)
            if self.early_stop and curr_perp + 3 > self.perplexities[-2]:
                self.logger.info("Doing early stopping, the model converged.")
                break
            # E step
            #1
            count_e_f = defaultdict(lambda: defaultdict(int))
            total_f = defaultdict(int)  # all expected alignment of f (target)
            #2
            count_alignment = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: 0)))
            )
            total_t_for_s = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: 0))
            )

            for source_sent, target_sent in tqdm(zip(self.source.data, self.target.data),
                                                 desc="Iterate all sents pairs", total=len(self.source.data)):
                # SENTENCE CONTEXT
                s_len = len(source_sent)
                t_len = len(target_sent)

                source_sent = [self.UNIQUE_NONE] + source_sent  # Adding Blank word in the beginning
                # compute normaliztion
                s_total = defaultdict(int)  # count
                for idx_s, s_w in enumerate(source_sent):
                    for idx_t, t_w in enumerate(target_sent):
                        idx_t += 1
                        s_total[s_w] += self.get_expected_prob(idx_s, idx_t, s_w, s_len, t_w, t_len)
                        assert idx_t <= len(target_sent)

                for idx_s, s_w in enumerate(source_sent):
                    for idx_t, t_w in enumerate(target_sent):
                        idx_t += 1
                        expected = self.get_expected_prob(idx_s, idx_t, s_w, s_len, t_w, t_len)
                        collected_count = expected / s_total[s_w]
                        # e given f
                        count_e_f[t_w][s_w] += collected_count
                        total_f[t_w] += collected_count
                        # alighmnet
                        count_alignment[idx_s][idx_t][s_len][t_len] += collected_count
                        total_t_for_s[idx_t][s_len][t_len] += collected_count

            # M step
            for t_w, t_w_count in tqdm(count_e_f.items(), desc='calculating vocab', total=len(count_e_f)):
                for s_w, val in t_w_count.items():
                    upd_prob = val / total_f[t_w]
                    self.prob_ef_expected_alignment[t_w][s_w] = upd_prob

            for s_idx, trg_indices in count_alignment.items():
                for t_idx, src_lengths in trg_indices.items():
                    for s_len, trg_sentence_lengths in src_lengths.items():
                        for t_len in trg_sentence_lengths:

                            upd_prob = count_alignment[s_idx][t_idx][s_len][t_len] / total_t_for_s[t_idx][s_len][t_len]
                            count_alignment[s_idx][t_idx][s_len][t_len] = upd_prob

    def get_expected_prob(self, idx_s, idx_t, s_w, source_len, t_w, target_len):
        return self.expected_alignment(s_w, t_w) * self.expected_distortion(idx_s, idx_t, source_len, target_len)

    def predict(self, source_sent, target_sent):
            res = []
            source_sent = [self.UNIQUE_NONE] + source_sent
            target_len = len(target_sent)
            source_len = len(source_sent)
            for t_idx, tw in enumerate(target_sent):
                # Initialize trg_word to align with the NULL token
                t_idx += 1
                best_prob = self.get_expected_prob(0, t_idx, self.UNIQUE_NONE, source_len, tw, target_len)  #TODO see if the index match
                probable_align = self.UNIQUE_NONE
                for s_idx, sw in enumerate(source_sent):
                    alignment = self.expected_alignment(sw, tw)
                    distortion = self.expected_distortion(s_idx, t_idx, target_len, source_len)
                    cur_val = alignment * distortion
                    if cur_val >= best_prob:
                        best_prob = cur_val
                        probable_align = s_idx - 1 # we added none now we take a step back
                    if probable_align != self.UNIQUE_NONE:
                        res.append(f"{t_idx}-{probable_align}")
            str_out = " ".join(res)
            str_out =str_out + "\n"
            return str_out


    def init_distortion_uniformly(self):
        distortion_table = defaultdict( # s_w
            lambda: defaultdict(        # t_w
                lambda: defaultdict(    # len_s
                    lambda: defaultdict(# len_t
                        lambda: int))))
        all_lengths = set()
        for source_sent, target_sent in zip(self.source.data, self.target.data):
            s_len = len(source_sent)  # We compute the sent with out the additional word
            t_len = len(target_sent)  # We compute the sent with out the additional word
            if (s_len, t_len) not in all_lengths:
                all_lengths.add((s_len, t_len))
                initial_prob = 1 / (s_len + 1)  # all the words  + None
                for s_idx in range(s_len + 1):
                    for t_idx in range(1, t_len + 1): #need to add a dummy word for len...
                        distortion_table[s_idx][t_idx][s_len][t_len] = initial_prob
        return distortion_table


    def expected_distortion(self, s_w, t_w, len_s, len_t):
        return self.distortion_table[s_w][t_w][len_s][len_t]

    def probability_e_f(self, source_sent, target_sent):
        source_len = len(source_sent)
        target_len = len(target_sent)
        source_sent = [self.UNIQUE_NONE] + source_sent
        p_e_f = 1
        for s_idx, sw in enumerate(source_sent):
            inner_sum = 0
            for t_idx, tw in enumerate(target_sent):
                t_idx += 1
                inner_sum += self.get_expected_prob(s_idx, t_idx, sw, source_len, tw, target_len)
            p_e_f = inner_sum * p_e_f
        p_e_f = p_e_f / (target_len ** source_len)
        return p_e_f

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Aligner model')
    parser.add_argument('-m', '--model', help='ibm model {1,2} ', default=None, type=int)
    parser.add_argument('-n', '--num_of_lines', help='Number of lines to use', default=None, type=int)
    parser.add_argument('-e', '--epochs', help='Number of epochs', default=5, type=int)
    # parser.add_argument('-d', '--diagonal', help='Prefer diagonal', action='store_true')
    parser.add_argument('-i', '--init_from_saved', help='init weights from saved pkl', action='store_true')
    parser.add_argument('-p', '--p2we', help='path to saved weights',  default=None)
    parser.add_argument('-s', '--early_stop', action='store_true')
    args = parser.parse_args()
    suf_fr = 'f'
    suf_en = 'e'
    suf_al = 'a'
    en = Lang(suf_en, args.num_of_lines)
    fr = Lang(suf_fr, args.num_of_lines)
    if args.model == 1:
        model = IbmModel1(en, fr, n_ep=args.epochs, init_from_saved_w=args.init_from_saved, early_stop=args.early_stop, path_to_probs=args.p2we )
    elif args.model == 2:
        model = IbmModel2(en, fr, n_ep=args.epochs, init_from_saved_w=args.init_from_saved, early_stop=args.early_stop)
    else:
        raise ValueError("model supports only 1 or 2")
    model.predict_all()


    


