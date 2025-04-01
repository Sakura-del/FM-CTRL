set -v
dataset="icews0515"
finding_mode="head"
device="cuda:0"
seed=42
relation_prediction_lr=1e-5
search_depth=3
search_depth=3


# "negative sampling"
python neg_sampling.py --dataset $dataset --finding_mode $finding_mode --training_mode train --neg_num 3 --seed $seed
python neg_sampling.py --dataset $dataset --finding_mode $finding_mode --training_mode valid --neg_num 50 --seed $seed
python neg_sampling.py --dataset $dataset --finding_mode $finding_mode --training_mode test --neg_num 50 --seed $seed

# "path finding"
python path_finding.py --dataset $dataset --finding_mode $finding_mode --training_mode train --npaths_ranking 3 --support_threshold $support_threshold --search_depth $search_depth
python path_finding.py --dataset $dataset --finding_mode $finding_mode --training_mode valid --npaths_ranking 3 --support_threshold $support_threshold --search_depth $search_depth
python path_finding.py --dataset $dataset --finding_mode $finding_mode --training_mode test --npaths_ranking 3 --search_depth $search_depth

# "relation prediction"
python entity_prediction.py --device $device --epochs 5 --batch_size 10 --dataset $dataset --learning_rate $relation_prediction_lr --neg_sample_num_train 3 --neg_sample_num_valid 50 --neg_sample_num_test 50 --mode $finding_mode --seed $seed --do_train --do_test

python relation_prediction.py --device cuda:0 --epochs 5 --batch_size 10 --dataset icews0515 --learning_rate 1e-5 --neg_sample_num_train 3 --neg_sample_num_valid 3 --neg_sample_num_test 50 --max_path_num 3 --mode head --seed 42 --do_train