mkdir data
mkdir data/Electronics_split_by_user
# Download dataset
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
# Unzip
#gunzip meta_Electronics.json.gz -c > data/meta_Electronics.json
#gunzip reviews_Electronics_5.json.gz -c > data/reviews_Electronics_5.json
# preprocessing
python script/process_data_split_by_user.py data/Electronics_split_by_user/ data/meta_Electronics.json data/reviews_Electronics_5.json
python script/local_aggretor.py data/Electronics_split_by_user/
python script/split_by_user.py data/Electronics_split_by_user/
python script/generate_voc.py data/Electronics_split_by_user/

# make necessary folders
mkdir data/Electronics_split_by_user/dnn_save_path/
mkdir data/Electronics_split_by_user/dnn_best_model/
mkdir data/Electronics_split_by_user/my_logs/
mkdir data/Electronics_split_by_user/loss_logs/
mkdir tmp/
