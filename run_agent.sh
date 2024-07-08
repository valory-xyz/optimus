kill_tm() {
    echo "Terminating tendermint..."
    if kill -0 "$1" 2>/dev/null; then
        kill "$subprocess_pid"
        wait "$subprocess_pid" 2>/dev/null
    fi
    echo "Tendermint terminated"
}

# Link kill_tm to the exit signal
trap kill_tm EXIT

# Remove previous agent if exists
if test -d optimism_agent; then
  echo "Removing previous agent build"
  rm -r optimism_agent
fi

# Remove empty directories to avoid wrong hashes
find . -empty -type d -delete

# Ensure hashes are updated
autonomy packages lock

# Fetch the agent
autonomy fetch --local --agent valory/optimism_agent
python scripts/aea-config-replace.py

# Copy and add the keys and issue certificates
cd optimism_agent
cp $PWD/../ethereum_private_key.txt .
autonomy add-key ethereum ethereum_private_key.txt
autonomy issue-certificates

# Run tendermint
rm -r ~/.tendermint
tendermint init > /dev/null 2>&1
tendermint node --proxy_app=tcp://127.0.0.1:26658 --rpc.laddr=tcp://127.0.0.1:26657 --p2p.laddr=tcp://0.0.0.0:26656 --p2p.seeds= --consensus.create_empty_blocks=true > /dev/null 2>&1 &
subprocess_pid=$!

# Run the agent
aea -s run