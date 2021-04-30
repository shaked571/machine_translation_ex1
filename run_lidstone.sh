for i in 1 2 4 6 8 10 12 14 16 18
do
python ibm1_em.py -m 1 -e 10 -lc -ld -ln ${i} -n 25000
done
for i in 1 2 4 6 8 10 12 14 16 18
do
python eval.py -e data/hansards.e -f  data/hansards.f  -a data/hansards.a -n 0 < prediction_IBM_Model1_epoch_10_use_null_True_lidstone_True_ld_n_${i}_num_of_line_25000_lower_case.txt
done

