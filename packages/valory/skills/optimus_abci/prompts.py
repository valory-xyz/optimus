PROMPT = """Based on the user-provided prompt, determine which protocols (from a predefined list) best address the user's request and decide the appropriate trading type based on the user's risk appetite.

- The user's prompt is: {USER_PROMPT}
- The previous trading type was: {PREVIOUS_TRADING_TYPE}
- The available trading types are: {TRADING_TYPES}
- The available protocols (with brief explanations) are: {AVAILABLE_PROTOCOLS}
- The last chosen composite score threshold was: {LAST_THRESHOLD}
- The last chosen protocols are: {PREVIOUS_SELECTED_PROTOCOLS}
- Thresholds values based on trading type: {THRESHOLDS}

A composite score is a single numerical value that represents the overall risk level of a trading strategy, taking into account various factors such as volatility, liquidity, and potential returns.

**Output Requirements**:

1. **Provide your answer as a valid JSON string ONLY. Do not include any additional text, explanations, or formatting.**

2. The JSON should have the following keys:
   - `'selected_protocols'`: A list (array) of protocol names relevant to the user prompt. If no relevant protocols are found, return an empty list `[]`.
   - `'trading_type'`: A string representing the chosen trading type (e.g., `'Risky'` or `'Balanced'`). If no suitable trading type is found, return an empty string `''`.
   - `'reasoning'`: A brief explanation of why the selected trading type and protocols were chosen, based on the user's risk appetite and the chosen composite score threshold and what it means and how has it changed from the last chosen threshold. If no relevant protocols or trading type are found, provide an explanation for why the user's request could not be understood or addressed. Mention that explicitly and state that we would be falling back to previous strategies and trading type. **The reasoning should be returned as HTML, with styling using <span> tags for emphasis and <p> tags for paragraphs.**

3. **Do not include any code snippets, code fences, or markdown formatting in your response.**
"""