
def read_allin(f_name):
    with open(f_name, encoding='utf-8') as f:
        lines = f.readlines()
    res = [l.split() for l in lines[:40]]
    return res


f_res = read_allin("combine/fr_en.txt")
e_res = read_allin("combine/en_fr.txt")
set_union = []
inters = []
for f, e in zip(f_res, e_res):
    set_union.append(list(set(e+f)))
    inters.append(list(set(e) & set(f)))
res_union = []
for s in set_union:
    r = " ".join(s)
    r += "\n"
    res_union.append(r)

with open("combine/res_union", mode='w') as f:
    f.writelines(res_union)

res_inters= []
for s in inters:
    r = " ".join(s)
    r += "\n"
    res_inters.append(r)

with open("combine/res_inters", mode='w') as f:
    f.writelines(res_inters)
