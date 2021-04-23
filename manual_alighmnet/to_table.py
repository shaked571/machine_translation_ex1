import re
with open("english.txt", encoding='utf-8') as f:
    english_file = f.readlines()
    eng = [re.findall(r"[\w\.']+|[,!?;]", e.strip()) for e in english_file]


import re
with open("hebrew.txt", encoding='utf-8') as f:
    heb_file = f.readlines()
    heb = [re.findall(r"[\w\.']+|[,!?;]", e.strip()) for e in heb_file]

zipped = []
longest = [max(i) for i in english_file]
for e, h in zip(eng, heb):
    zipped.append(e)
    zipped.append([])
    zipped.append(h)

print(zipped)
import pandas as pd
import numpy as np
# pd.DataFrame(zipped).to_csv("table.csv", encoding="utf-8")

with open("table2.txt", mode='w', encoding='utf-8') as f:
    pd.DataFrame(zipped).to_string(f)
#

"""
given sent t_1, ...,  t_n  
1. sample a length
2. for each target position:
    2.1 sample an alignment pos to the source
    2.2 Sample a word translation given this alignment
     # PIE (p(alignment| source_sent_len) * 
     #      p(foreign_word_in_position_i| sample word_translation))

Need to learn:
- sent len distribution
- Alignment distributions
- Word translation distributions
initialize t(e|f) uniformly
 do until convergence
   set count(e|f) to 0 for all e,f
   set total(f) to 0 for all f
   for all sentence pairs (e_s,f_s)
     set total_s(e) = 0 for all e
     for all words e in e_s
       for all words f in f_s
         total_s(e) += t(e|f)
     for all words e in e_s
       for all words f in f_s
         count(e|f) += t(e|f) / total_s(e)
         total(f)   += t(e|f) / total_s(e)
   for all f
     for all e
       t(e|f) = count(e|f) / total(f)
"""
