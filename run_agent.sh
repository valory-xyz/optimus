cleanup() {
    echo "Terminating tendermint..."
    if kill -0 "$tm_subprocess_pid" 2>/dev/null; then
        kill "$tm_subprocess_pid"
        wait "$tm_subprocess_pid" 2>/dev/null
    fi
    echo "Tendermint terminated"

    if [ "$USE_TEST_API" = "true" ]; then
      echo "Terminating test pool API..."
      if kill -0 "$api_subprocess_pid" 2>/dev/null; then
        kill "$api_subprocess_pid"
        wait "$api_subprocess_pid" 2>/dev/null
      fi
      echo "Pool API terminated"
    fi
}

# Load env vars
source .env
repo_path=$PWD

# Link cleanup to the exit signal
trap cleanup EXIT

# Remove previous agent if exists
if test -d optimus; then
  echo "Removing previous agent build"
  rm -r optimus
fi

# Remove empty directories to avoid wrong hashes
find . -empty -type d -delete

# Ensure hashes are updated
autonomy packages lock

# Fetch the agent
autonomy fetch --local --agent valory/optimus
python scripts/aea-config-replace.py

# Copy and add the keys and issue certificates
cd optimus
cp $PWD/../ethereum_private_key.txt .
autonomy add-key ethereum ethereum_private_key.txt
autonomy issue-certificates

# Run tendermint
rm -r ~/.tendermint
tendermint init > /dev/null 2>&1
echo "Starting Tendermint..."
tendermint node --proxy_app=tcp://127.0.0.1:26658 --rpc.laddr=tcp://127.0.0.1:26657 --p2p.laddr=tcp://0.0.0.0:26656 --p2p.seeds= --consensus.create_empty_blocks=true > /dev/null 2>&1 &
tm_subprocess_pid=$!

# Run testing API
if [ "$USE_TEST_API" = "true" ]; then
  echo "Starting test pool API..."
  python $repo_path/scripts/run_merkle_api.py > /dev/null 2>&1 &
  api_subprocess_pid=$!
fi

# Run the agent
aea -s run