PROMPT = """Based on the user-provided prompt, determine which strategies
(from a predefined list) best address the user's request. - The user's prompt
is- {USER_PROMPT} - The set of available strategies (with brief explanations)
is {STRATEGIES} 1. asset_lending - Identifies the highest-yielding aggregators
for lending assets on Sturdy Finance. - Focuses on interest-free borrowing
and stable deposit yields through Sturdy's secure lending protocol. 2. balancer_pool_search
- Finds the highest-yielding pools on Balancer. - Evaluates pool fees, liquidity,
and potential earnings. Output Requirements- 1. Provide your answer ONLY in
JSON format (WITHOUT extra explanation). 2. The JSON must have a single key
'strategies'. 3. The value for 'strategies' should be a list (array) of strategy
names relevant to the user prompt. 4. If no relevant strategies exist, return
an empty list."""