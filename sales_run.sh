#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "Starting server"
python sales_server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 4`; do
    echo "Starting client $i"
    python sales_client.py $i &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait