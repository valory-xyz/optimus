import statistics
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

class BalancerLiquidityAnalyzer:
    def __init__(self, api_url='https://api-v3.balancer.fi', price_impact=0.01):
        """
        Initialize Balancer Liquidity Analyzer
        
        :param api_url: GraphQL API endpoint
        :param price_impact: Standardized price impact (default 1%)
        """
        self.api_url = api_url
        self.price_impact = price_impact
        self.transport = RequestsHTTPTransport(url=self.api_url, verify=True, retries=3)
        self.client = Client(transport=self.transport, fetch_schema_from_transport=False)
    
    def create_query(self, pool_id, range='NINETY_DAYS', chain='MAINNET'):
        """
        Create GraphQL query for fetching pool snapshots
        
        :param pool_id: Balancer pool ID
        :param range: Time range for snapshots
        :param chain: Blockchain network
        :return: GraphQL query
        """
        return gql(f'''
        query GetLiquidityMetrics {{
          poolGetSnapshots(
            id: "{pool_id}",
            range: {range},
            chain: {chain}
          ) {{
            totalLiquidity
            volume24h
            timestamp
          }}
        }}
        ''')
    
    def fetch_liquidity_metrics(self, pool_id):
        """
        Fetch liquidity metrics for a specific pool
        
        :param pool_id: Balancer pool ID
        :return: List of pool snapshots
        """
        try:
            query = self.create_query(pool_id)
            response = self.client.execute(query)
            return response['poolGetSnapshots']
        except Exception as e:
            print(f"An error occurred while fetching metrics: {e}")
            return []
    
    def calculate_liquidity_metrics(self, pool_snapshots):
        """
        Calculate liquidity metrics for Balancer pool
        
        :param pool_snapshots: List of pool snapshots
        :return: Dictionary of calculated metrics
        """
        if not pool_snapshots:
            raise ValueError("No pool snapshots provided")
        
        # Calculate average metrics
        avg_tvl = statistics.mean(float(snapshot['totalLiquidity']) for snapshot in pool_snapshots)
        avg_volume = statistics.mean(float(snapshot.get('volume24h', 0)) for snapshot in pool_snapshots)
        
        # Depth Score Calculation
        depth_score = (avg_tvl * avg_volume) / (self.price_impact * 100)
        
        # Liquidity Risk Multiplier
        liquidity_risk_multiplier = max(0, 1 - (1 / depth_score)) if depth_score != 0 else 0
        
        # Maximum Position Size
        max_position_size = 50 * (avg_tvl * liquidity_risk_multiplier) / 100
        
        # Prepare results
        results = {
            'Average TVL': avg_tvl,
            'Average Daily Volume': avg_volume,
            'Depth Score': depth_score,
            'Liquidity Risk Multiplier': liquidity_risk_multiplier,
            'Maximum Position Size': max_position_size,
            'Meets Depth Score Threshold': depth_score > 50
        }
        
        return results
    
    def analyze_pool_data(self, pool_id):
        """
        Comprehensive analysis of pool liquidity metrics
        
        :param pool_id: Balancer pool ID
        :return: Detailed analysis report
        """
        # Fetch pool snapshots
        pool_snapshots = self.fetch_liquidity_metrics(pool_id)
        
        # Calculate metrics
        metrics = self.calculate_liquidity_metrics(pool_snapshots)
        
        # Generate detailed report
        print("Balancer Pool Liquidity Analysis Report")
        print("-" * 50)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # Risk Assessment
        risk_assessment = []
        if metrics['Depth Score'] > 50:
            risk_assessment.append("✓ Depth Score meets threshold")
        else:
            risk_assessment.append("✗ Depth Score below recommended threshold")
        
        if metrics['Liquidity Risk Multiplier'] > 0.5:
            risk_assessment.append("✓ Low Liquidity Risk")
        else:
            risk_assessment.append("⚠ Moderate to High Liquidity Risk")
        
        print("\nRisk Assessment:")
        for assessment in risk_assessment:
            print(assessment)
        
        return metrics

def main():
    # Balancer pool ID (replace with the actual pool ID you're analyzing)
    pool_id = "0x5c6ee304399dbdb9c8ef030ab642b10820db8f56000200000000000000000014"
    
    # Initialize the analyzer
    analyzer = BalancerLiquidityAnalyzer()
    
    # Analyze the pool data
    results = analyzer.analyze_pool_data(pool_id)
    print(f"result of static:{results}")

if __name__ == "__main__":
    main()