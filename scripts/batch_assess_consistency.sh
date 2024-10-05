

for anchor_num in 4000 10000 20000; do
    for knn in 1 3 5 10 50 100; do
        python utils/assess_anchor_embedding_last_hidden_new.py $anchor_num $knn
    done
done