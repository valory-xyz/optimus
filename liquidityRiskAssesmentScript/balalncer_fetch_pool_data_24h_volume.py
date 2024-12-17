from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

# Set up the GraphQL client
api_url = 'https://api-v3.balancer.fi'
transport = RequestsHTTPTransport(url=api_url, verify=True, retries=3)
client = Client(transport=transport, fetch_schema_from_transport=False)

# GraphQL query
query = gql('''
query GetLiquidityMetrics {
  poolGetSnapshots(
    id: "0x5c6ee304399dbdb9c8ef030ab642b10820db8f56000200000000000000000014",
    range: NINETY_DAYS,
    chain: MAINNET
  ) {
    totalLiquidity
    volume24h
    timestamp
  }
}
''')

def fetch_liquidity_metrics():
    try:
        response = client.execute(query)
        for snapshot in response['poolGetSnapshots']:
            print(f"Timestamp: {snapshot['timestamp']}, Total Liquidity: {snapshot['totalLiquidity']}, Volume (24h): {snapshot['volume24h']}")
    except Exception as e:
        print(f"An error occurred: {e}")

fetch_liquidity_metrics()
