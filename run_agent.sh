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

# Run the agent
aea -s run