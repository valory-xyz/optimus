"""Module for handling prompts in the Optimus ABCI skill."""

PROMPT = """Analyze the user prompt to determine protocols and risk tolerance.

- User prompt: {USER_PROMPT}
- Previous trading type: {PREVIOUS_TRADING_TYPE}
- Available trading types: {TRADING_TYPES}
- Available protocols: {AVAILABLE_PROTOCOLS}
- Last threshold: {LAST_THRESHOLD}
- Previous protocols: {PREVIOUS_SELECTED_PROTOCOLS}
- Threshold values: {THRESHOLDS}

Analyze risk sentiment in the user's language and estimate their maximum acceptable loss percentage:
- Conservative language ("safe", "minimize risk"): 1-5%
- Moderate language ("balanced", "stable"): 6-10%
- Growth-focused ("higher returns", "some risk"): 11-15%
- Aggressive ("maximize", "high returns"): 16-25%
- Very aggressive ("maximum returns", "big risks"): 26-30%
Default to 10% if unclear.

Return JSON with these keys:
- 'selected_protocols': Array of relevant protocol names
- 'trading_type': String ('risky' or 'balanced')
- 'max_loss_percentage': Number between 1-30 representing risk tolerance
- 'reasoning': HTML explanation of selections, inferred risk level, and effects

Only return valid JSON. No code snippets or markdown.
"""