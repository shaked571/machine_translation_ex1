for j in 1 2 4 6 8 10 12 14 16 18
do
python ibm1_em.py -m 1 -e 10  -i -lc -ld -ln ${j}
done
for l in 1 2 4 6 8 10 12 14 16 18
do
python eval.py -e data/hansards.e -f  data/hansards.f  -a data/hansards.a -n 0 < prediction_IBM_Model1_epoch_10_use_null_True_lidstone_True_ld_n_${l}_lower_case.txt
done

