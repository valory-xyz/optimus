from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

# Set up the GraphQL client
api_url = 'https://api-v3.balancer.fi'  # Ensure this is the correct API endpoint
transport = RequestsHTTPTransport(url=api_url, verify=True, retries=3)
client = Client(transport=transport, fetch_schema_from_transport=False)

# Define the GraphQL query 
query = gql('''
query GetPoolSnapshots {
  poolGetSnapshots(id: "0x5c6ee304399dbdb9c8ef030ab642b10820db8f56000200000000000000000014", range: NINETY_DAYS, chain: MAINNET) {
    id
    totalLiquidity
    totalSwapVolume
    timestamp
  }
}
''')

def fetch_pool_snapshots():
    try:
        response = client.execute(query)
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")

fetch_pool_snapshots()
