// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract MockBPT is ERC20 {
    address public vault;
    
    constructor(string memory name, string memory symbol) ERC20(name, symbol) {
        vault = msg.sender;
    }
    
    function mint(address to, uint256 amount) external {
        require(msg.sender == vault, "Only vault");
        _mint(to, amount);
    }
}
