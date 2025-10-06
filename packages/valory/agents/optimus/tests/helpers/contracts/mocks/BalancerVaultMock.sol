// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "./MockBPT.sol";

/**
 * @dev Minimal Balancer Vault mock for testing
 */
contract BalancerVaultMock {
    
    event PoolJoined(
        bytes32 indexed poolId,
        address indexed sender,
        address indexed recipient,
        uint256 bptOut
    );
    
    mapping(bytes32 => address) public poolIdToAddress;
    
    function registerPool(bytes32 poolId, address poolAddress) external {
        poolIdToAddress[poolId] = poolAddress;
    }
    
    function joinPool(
        bytes32 poolId,
        address sender,
        address recipient,
        JoinPoolRequest memory request
    ) external {
        require(poolIdToAddress[poolId] != address(0), "Pool not registered");
        
        // Transfer tokens from sender
        for (uint256 i = 0; i < request.assets.length; i++) {
            IERC20(request.assets[i]).transferFrom(
                sender,
                address(this),
                request.maxAmountsIn[i]
            );
        }
        
        // Calculate BPT to mint (simplified: sum of amounts)
        uint256 bptOut = request.maxAmountsIn[0] + request.maxAmountsIn[1];
        
        // Mint LP tokens to recipient
        address poolAddress = poolIdToAddress[poolId];
        MockBPT(poolAddress).mint(recipient, bptOut);
        
        emit PoolJoined(poolId, sender, recipient, bptOut);
    }
    
    struct JoinPoolRequest {
        address[] assets;
        uint256[] maxAmountsIn;
        bytes userData;
        bool fromInternalBalance;
    }
}
