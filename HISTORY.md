# Release History - `optimus`

## v0.1.0 - [Release Date: 2024-09-06] 

- Feat/uniswap
- Feat/claim reward
- fix: claim rewards

## v0.2.0 - [Release Date: 2024-09-09]

- Fix/unlocking staking rewards 

## v0.2.1 - [Release Date: 2024-09-18]

- Fix/staking 
- Fix/include all campaigns 

## v0.2.2 - [Release Date: 2024-09-20]

- Fix/vanity tx 

## v0.2.3 

- fix: resolve CI failures  
- chore: add release.yaml 

## v0.2.4

- chore: bump to autonomy@v0.16.1 
- fix: remove gas overrides  

## v0.2.5 

- Fix/swap profitability 
- Implements logic in the Optimus agent to proceed with a bridge or swap transaction only if the relayer fee is less than 2% of the amount being bridged or swapped, and the gas fees are less than 25% of the amount being traded.

## v0.2.6

- Fix/route selection 
- fix: update url 
- fix: add check for zero address 
- fix: remove retries for status check 
- Enhanced the reliability and efficiency of token swaps and bridges.
- Introduced fallback mechanisms for failed routes and cost-efficiency checks to control gas and swap/bridge fees.
- Optimized route selection to reduce complexity and retries, improving transaction success and cost-effectiveness.

## v0.2.7 

- adds gas cost tracker  

## v0.2.8

- fix: make allowed chains configurable 


