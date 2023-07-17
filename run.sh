#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/


BATCH_SIZE=32
NUM_CLIENTS=3
# DATA_OUTPUT='./data_tmp.pt'
DATA_OUTPUT="./change_data_tmp.pt"
# TRAIN_DATA='../data/tmp_train_data.csv'
TRAIN_DATA='../change_data/tmp_train_data.csv'


echo "Starting server"
python3 Server.py  -n $NUM_CLIENTS -do $DATA_OUTPUT -f $TRAIN_DATA &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 1 2`; do
    echo "Starting client $i"
    sleep 1 
    python3 Client.py ${i} -d $DATA_OUTPUT &
done


# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

# # Test VFL system
python3 compute_test_metrics.py -n $NUM_CLIENTS -d $DATA_OUTPUT

python3 generate_test_probabilities.py -s "onplateu-max-f1"

