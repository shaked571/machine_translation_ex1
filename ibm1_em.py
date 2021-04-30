import abc
import argparse
import math
import sys
from datetime import datetime
from typing import List
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

    def __init__(self, suf, number_of_lines=None, do_lowe_case=False):
        self.suf = suf
        self.logger = logging.getLogger(f"lang-{self.suf}")
        self.do_lower_case = do_lowe_case
        self.number_of_lines = number_of_lines
        self.data = self.read_file(self.base_name + self.suf)
        self.voc = Counter(self.flatten(self.data))
        self.unique = len(self.voc)
        self.w_index = {w: i for i, w in enumerate(self.voc)}

    def read_file(self, f_name: str) -> List[List[str]]:
        with open(f_name, encoding='utf-8') as f:
            lines = f.readlines()

        if self.do_lower_case:
            res = [[w.lower() for w in line.split()] for line in lines]
        else:
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

    def __init__(self, source: Lang, target: Lang, n_ep,early_stop, model_name, change_direction,dont_use_null, lidstone ):
        self.model_name = model_name
        self.n_ep: int = n_ep
        self.early_stop: bool = early_stop
        self.prob_ef_expected_alignment = None
        self.saved_weight_fn = None
        self.dont_use_null:bool =dont_use_null
        self.lidstone:bool =lidstone
        self.change_direction = change_direction
        if self.change_direction:
            self.target: Lang =source
            self.source: Lang = target
        else:
            self.target: Lang = target
            self.source: Lang = source
        self.saved_weight_fn_model_1 = 'ibm1_p.pkl'
        self.logger = logging.getLogger(self.model_name)
        self.logger.info(f"Start model: {self.model_name}")
        self.logger.info(f"would num of epoch: {self.n_ep}")
        self.logger.info(f"Start with early stop: {self.early_stop}")
        self.logger.info(f"Start with lidstone: {self.lidstone}")
        self.logger.info(f"Start with using null: {not self.dont_use_null}")
        self.logger.info(f"Start with change direction: {self.change_direction}")
        self.extra_parameters = ''


    def expected_alignment(self, s_w, t_w):
        return self.prob_ef_expected_alignment[s_w][t_w]


    def add_special_null(self, lang):
        if self.dont_use_null:
            return lang
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

    def predict_all(self, extra_info):
        res = []
        for source_sent, target_sent in tqdm(zip(self.source.data, self.target.data),
                                             desc="predicting sentences", total=len(self.source.data)):
            res.append(self.predict(source_sent, target_sent))
        f_name = f"prediction_{self.model_name}_epoch_{self.n_ep}_use_null_{not self.dont_use_null}_lidstone_{self.lidstone}{extra_info}.txt"
        self.logger.info(f"writing to: {f_name}")
        with open(f_name, mode='w') as f:
            f.writelines(res)


    @abc.abstractmethod
    def predict(self, source_sent, target_sent):
        pass

    @abc.abstractmethod
    def probability_e_f(self, source_sent, target_sent):
        pass

    def save_alighnment(self) -> None:
        self.logger.info("Saving probs...")
        for k, v in self.prob_ef_expected_alignment.items():
            self.prob_ef_expected_alignment[k] = dict(v)
        self.prob_ef_expected_alignment = dict(self.prob_ef_expected_alignment)
        with open(self.saved_weight_fn, 'wb') as f:
            pickle.dump(dict(self.prob_ef_expected_alignment), f, pickle.HIGHEST_PROTOCOL)

    def add_line(self, res, t_idx, probable_align):
        if self.change_direction:
            res.append(f"{t_idx}-{probable_align}")
        else:
            res.append(f"{probable_align}-{t_idx}")


class IbmModel1(IbmModel):

    def __init__(self, source: Lang, target: Lang, n_ep=100, early_stop=True, init_from_saved_w=False, path_to_probs=None, saved_weight_fn=None,
                 change_direction=False,dont_use_null=False, lidstone=False):
        super(IbmModel1, self).__init__(source, target, n_ep, early_stop,'IBM_Model1',change_direction,dont_use_null,lidstone )
        self.saved_weight_fn = 'ibm1_p.pkl'
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
            total_f = defaultdict(int)  # a dict from source (french) to  target (english) {f:e}
            for source_sent, target_sent in tqdm(zip(self.source.data, self.target.data),
                                                 desc="Iterate all sents pairs", total=len(self.source.data)):
                # SENTENCE CONTEXT
                if not self.dont_use_null:
                    source_sent = [self.UNIQUE_NONE] + source_sent  # Adding Blank word in the beginning

                s_total = defaultdict(int)  # count
                if self.lidstone:
                    self.lid_prob_ef_expected_alignment()

                for t_w in target_sent:
                    for s_w in source_sent:
                        s_total[t_w] += self.expected_alignment(s_w, t_w)

                for t_w in target_sent:
                    for s_w in source_sent:
                        expected = self.expected_alignment(s_w, t_w)
                        collected_count = expected / s_total[t_w]
                        count_e_f[s_w][t_w] += collected_count
                        total_f[s_w] += collected_count
            # M step
            for s_w, s_w_count in tqdm(count_e_f.items(), desc='calculating vocab', total=len(count_e_f)):
                for t_w, val in s_w_count.items():
                    upd_prob = val / total_f[s_w]
                    self.prob_ef_expected_alignment[s_w][t_w] = upd_prob





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
            if self.dont_use_null:
                best_prob = 0
            else:
                best_prob = self.expected_alignment(self.UNIQUE_NONE, tw)
            probable_align = self.UNIQUE_NONE
            for s_idx, sw in enumerate(source_sent):
                curr_prob = self.expected_alignment(sw, tw)
                if curr_prob >= best_prob:
                    best_prob = curr_prob
                    probable_align = s_idx
            if probable_align != self.UNIQUE_NONE:
                self.add_line(res, t_idx, probable_align)
        str_out = " ".join(res)
        str_out =str_out + "\n"
        return str_out


    def save_probs(self):
        self.save_alighnment()

    def lid_prob_ef_expected_alignment(self):
        pass


#1. Add lidstone smoothing
#2. Adding null to the source sentence
#My ideas:
#2. Do a normalization over the vocabulary - 'lower' to make unique.
# Add extra null words?

class IbmModel2(IbmModel):

    saved_distortion_fn = 'ibm2_distortion.pkl'
    def __init__(self, source: Lang, target: Lang, n_ep=100, early_stop=True, init_from_saved_w=False,
                 path_to_probs=None, saved_weight_fn=None, saved_distortion_fn=None,use_model_1=False,
                 change_direction=False,dont_use_null=False, lidstone=False):
        super().__init__(source, target, n_ep, early_stop,  "IBM_Model2", change_direction,dont_use_null, lidstone)
        self.saved_weight_fn = 'ibm2_p.pkl'
        self.source: Lang = self.add_special_null(self.source)
        #Need to add length of both sentences
        self.n_ep: int = n_ep
        if saved_weight_fn is not None:
            self.saved_weight_fn = saved_weight_fn
        if saved_distortion_fn is not None:
            self.saved_distortion_fn = saved_distortion_fn

        if init_from_saved_w:
            self.logger.info("loading...")
            self.prob_ef_expected_alignment = self.load_probs(path_to_probs, self.saved_weight_fn)
            self.distortion_table = self.load_probs(path_to_probs, self.saved_distortion_fn)

        else:
            if use_model_1:
                self.prob_ef_expected_alignment = self.load_probs(path_to_probs, self.saved_weight_fn_model_1 )
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
        self.save_alighnment()
        self.save_distortion()

    def save_distortion(self):
        for k, v in self.distortion_table.items():
            self.distortion_table[k] = dict(v)
            for k1, v1 in self.distortion_table[k].items():
                self.distortion_table[k][k1] = dict(v1)
                for k2, v2 in self.distortion_table[k][k1].items():
                    self.distortion_table[k][k1][k2] = dict(v2)
        self.distortion_table = dict(self.distortion_table)
        with open(self.saved_distortion_fn, 'wb') as f:
            pickle.dump(self.distortion_table, f, pickle.HIGHEST_PROTOCOL)


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
            total_f = defaultdict(int)  # a dict from source (french) to  target (english) {f:e}
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
                if not self.dont_use_null:
                    source_sent = [self.UNIQUE_NONE] + source_sent  # Adding Blank word in the beginning
                # compute normaliztion
                delta_k = defaultdict(int)  # count
                for idx_t, t_w in enumerate(target_sent):
                    for idx_s, s_w in enumerate(source_sent):
                        delta_k[t_w] += self.get_expected_prob(idx_s, idx_t, s_w, s_len, t_w, t_len)

                for idx_t, t_w in enumerate(target_sent):  # 1,2,3
                    for idx_s, s_w in enumerate(source_sent):
                        expected = self.get_expected_prob(idx_s, idx_t, s_w, s_len, t_w, t_len)
                        collected_count = expected / delta_k[t_w]
                        # e given f
                        count_e_f[s_w][t_w] += collected_count
                        total_f[s_w] += collected_count
                        # alighmnet
                        count_alignment[idx_t][s_len][t_len][idx_s] += collected_count
                        total_t_for_s[idx_t][s_len][t_len] += collected_count

            # M step
            for s_w, s_w_count in tqdm(count_e_f.items(), desc='calculating vocab', total=len(count_e_f)):
                for t_w, val in s_w_count.items():
                    upd_prob = val / total_f[s_w]
                    self.prob_ef_expected_alignment[s_w][t_w] = upd_prob

            for idx_t, src_lengths in count_alignment.items():
                for s_len, trg_sentence_lengths in src_lengths.items():
                    for t_len, s_indices in trg_sentence_lengths.items():
                        for idx_s in s_indices:
                            upd_prob = count_alignment[idx_t][s_len][t_len][idx_s] / total_t_for_s[idx_t][s_len][t_len]
                            count_alignment[idx_t][s_len][t_len][idx_s] = upd_prob

    def expected_distortion(self, idx_s, idx_t, len_s, len_t):
        return self.distortion_table[idx_t][len_s][len_t][idx_s]

    def get_expected_prob(self, idx_s, idx_t, s_w, source_len, t_w, target_len):
        return self.expected_alignment(s_w, t_w) * self.expected_distortion(idx_s, idx_t, source_len, target_len)

    def predict(self, source_sent, target_sent):
            res = []
            target_len = len(target_sent)
            source_len = len(source_sent)
            for t_idx, t_w in enumerate(target_sent):
                # Initialize trg_word to align with the NULL token
                if self.dont_use_null:
                    best_prob = 0
                else:
                    best_prob = self.get_expected_prob(0, t_idx, self.UNIQUE_NONE, source_len, t_w, target_len)
                probable_align = self.UNIQUE_NONE
                for idx_s, s_w in enumerate(source_sent):
                    if self.dont_use_null:
                        cur_val = self.get_expected_prob(idx_s, t_idx, s_w, source_len, t_w, target_len)
                    else:
                        cur_val = self.get_expected_prob(idx_s+1, t_idx, s_w, source_len, t_w, target_len)
                    if cur_val >= best_prob:
                        best_prob = cur_val
                        probable_align = idx_s
                if probable_align != self.UNIQUE_NONE:
                    self.add_line(res, t_idx, probable_align)
            str_out = " ".join(res)
            str_out =str_out + "\n"
            return str_out

    def init_distortion_uniformly(self):
        distortion_table = defaultdict( # s_idx
            lambda: defaultdict(        # t_idx
                lambda: defaultdict(    # len_s
                    lambda: defaultdict(# len_t
                        lambda: int))))
        all_lengths = set()
        for source_sent, target_sent in zip(self.source.data, self.target.data):
            len_s = len(source_sent)  # We compute the sent with out the additional word
            len_t = len(target_sent)  # We compute the sent with out the additional word
            if (len_s, len_t) not in all_lengths:
                all_lengths.add((len_s, len_t))
                if self.dont_use_null:
                    initial_prob = 1 / (len_s)  # all the words  + Nonef
                    for idx_s in range(len_s):
                        for idx_t in range(len_t):
                            distortion_table[idx_t][len_s][len_t][idx_s] = initial_prob
                else:
                    initial_prob = 1 / (len_s + 1)  # all the words  + Nonefv
                    for idx_s in range(len_s + 1):
                        for idx_t in range(len_t):
                            distortion_table[idx_t][len_s][len_t][idx_s] = initial_prob

        return distortion_table


    def probability_e_f(self, source_sent, target_sent):
        source_len = len(source_sent)
        target_len = len(target_sent)
        if not self.dont_use_null:
            source_sent = [self.UNIQUE_NONE] + source_sent
        p_e_f = 1
        for s_idx, sw in enumerate(source_sent):
            inner_sum = 0
            for t_idx, tw in enumerate(target_sent):
                inner_sum += self.get_expected_prob(s_idx, t_idx, sw, source_len, tw, target_len)
            p_e_f = inner_sum * p_e_f
        p_e_f = p_e_f / (target_len ** source_len)
        return p_e_f

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aligner model')
    parser.add_argument('-m', '--model', help='ibm model {1,2} ', default=None, type=int)
    parser.add_argument('-n', '--num_of_lines', help='Number of lines to use', default=None, type=int)
    parser.add_argument('-e', '--epochs', help='Number of epochs', default=5, type=int)
    parser.add_argument('-lc', '--lower_case', help='lower case all token', action='store_true')
    parser.add_argument('-i', '--init_from_saved', help='init weights from saved pkl', action='store_true')
    parser.add_argument('-p', '--p2we', help='path to saved weights',  default=None)
    parser.add_argument('-o', '--use_model_1', action='store_true')
    parser.add_argument('-s', '--early_stop', action='store_true')
    parser.add_argument('-dn', '--dont_use_null', action='store_true') #TODO
    parser.add_argument('-cd', '--change_direction', help='switch target and source' ,action='store_true') #TODO
    parser.add_argument('-ld', '--lidstone', help='smoothing using lidstone' ,action='store_true')#TODO
    args = parser.parse_args()
    suf_fr = 'f'
    suf_en = 'e'
    suf_al = 'a'
    en = Lang(suf_en, args.num_of_lines, args.lower_case)
    fr = Lang(suf_fr, args.num_of_lines, args.lower_case)
    if args.model == 1:
        model = IbmModel1(fr, en, n_ep=args.epochs, init_from_saved_w=args.init_from_saved, early_stop=args.early_stop, path_to_probs=args.p2we,
                          change_direction=args.change_direction,dont_use_null=args.dont_use_null, lidstone=args.lidstone)
    elif args.model == 2:
        model = IbmModel2(fr, en, n_ep=args.epochs, init_from_saved_w=args.init_from_saved, early_stop=args.early_stop, path_to_probs=args.p2we, use_model_1=args.use_model_1,
                          change_direction=args.change_direction,dont_use_null=args.dont_use_null, lidstone=args.lidstone)
    else:
        raise ValueError("model supports only 1 or 2")
    extra_info = ''
    if args.num_of_lines:
        extra_info += f'_num_of_line_{args.num_of_lines}'
    if args.lower_case:
        extra_info += f'_lower_case'
    model.predict_all(extra_info)


    


