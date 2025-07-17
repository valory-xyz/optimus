"""Module for handling prompts in the Optimus ABCI skill."""

PROMPT = """Analyze the user prompt to determine protocols and risk tolerance.

- User prompt: {USER_PROMPT}
- Previous trading type: {PREVIOUS_TRADING_TYPE}
- Available trading types: {TRADING_TYPES}
- Available protocols: {AVAILABLE_PROTOCOLS}
- Last threshold: {LAST_THRESHOLD}
- Previous protocols: {PREVIOUS_SELECTED_PROTOCOLS}
- Threshold values: {THRESHOLDS}

IMPORTANT: You must use ONLY these exact protocol names (case-sensitive):
- "balancerPool" (for Balancer protocol)
- "uniswapV3" (for Uniswap V3 protocol)
- "velodrome" (for Velodrome protocol)
- "sturdy" (for Sturdy lending protocol)

DO NOT use abbreviations like "uni", "velo", etc. Use the exact names above.

CRITICAL INSTRUCTIONS FOR PROTOCOL SELECTION:
1. If the user explicitly specifies to use some protocol:
   - Use just that protocol
2. If the user says "remove [protocol]" or "remove [protocol] from pool":
   - Remove ONLY the specified protocol from the previous protocols
   - Keep all other previous protocols unchanged
   - If the protocol to remove is not in the previous protocols, keep previous protocols unchanged
3. If the user says "add [protocol]" or "add [protocol] to pool":
   - Add the specified protocol to the previous protocols
   - Keep all other previous protocols unchanged
4. If the user specifies a complete new list of protocols:
   - Use the exact protocols they specify
5. If the user doesn't specify protocols:
   - Keep the previous protocols unchanged

Analyze risk sentiment in the user's language and estimate their maximum acceptable loss percentage:
- Conservative language ("safe", "minimize risk"): 1-5%
- Moderate language ("balanced", "stable"): 6-10%
- Growth-focused ("higher returns", "some risk"): 11-15%
- Aggressive ("maximize", "high returns"): 16-25%
- Very aggressive ("maximum returns", "big risks"): 26-30%
Default to 10% if unclear.

Return JSON with these keys:
- 'selected_protocols': Array of valid protocol names (ONLY use: "balancerPool", "uniswapV3", "velodrome", "sturdy")
- 'trading_type': String ('risky' or 'balanced')
- 'max_loss_percentage': Number between 1-30 representing risk tolerance
- 'reasoning': HTML explanation of selections, inferred risk level, and effects

Only return valid JSON. No code snippets or markdown.
"""
