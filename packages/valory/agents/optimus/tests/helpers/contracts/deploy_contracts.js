const hre = require("hardhat");

// Addresses
const SAFE_ADDRESS = "0x1234567890123456789012345678901234567890";
const MULTISEND_ADDRESS = "0x40A2aCCbd92BCA938b02010E17A5b8929b49130D";

async function main() {
  console.log("Deploying contracts to local Hardhat...");
  
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deployer:", deployer.address);
  
  // 1. Deploy Mock USDC (6 decimals)
  const USDC = await hre.ethers.getContractFactory("MockERC20");
  const usdc = await USDC.deploy("USD Coin", "USDC", 6);
  await usdc.waitForDeployment();
  console.log("USDC deployed to:", await usdc.getAddress());
  
  // 2. Deploy Mock USDT (6 decimals)
  const USDT = await hre.ethers.getContractFactory("MockERC20");
  const usdt = await USDT.deploy("Tether USD", "USDT", 6);
  await usdt.waitForDeployment();
  console.log("USDT deployed to:", await usdt.getAddress());
  
  // 3. Mint tokens to Safe
  const usdcAmount = hre.ethers.parseUnits("10", 6);
  const usdtAmount = hre.ethers.parseUnits("10", 6);
  
  await usdc.mint(SAFE_ADDRESS, usdcAmount);
  await usdt.mint(SAFE_ADDRESS, usdtAmount);
  console.log("Minted 10 USDC and 10 USDT to Safe");
  
  // 4. Deploy Balancer Vault Mock
  const Vault = await hre.ethers.getContractFactory("BalancerVaultMock");
  const vault = await Vault.deploy();
  await vault.waitForDeployment();
  console.log("BalancerVaultMock deployed to:", await vault.getAddress());
  
  // 5. Deploy Mock BPT (LP token)
  const BPT = await hre.ethers.getContractFactory("MockBPT");
  const bpt = await BPT.deploy("Balancer USDC-USDT LP", "B-USDC-USDT");
  await bpt.waitForDeployment();
  console.log("MockBPT deployed to:", await bpt.getAddress());
  
  // 6. Register pool in vault
  const poolId = "0x7b50775383d3d6f0215a8f290f2c9e2eebbeceb20000000000000000000000fe";
  await vault.registerPool(poolId, await bpt.getAddress());
  console.log("Pool registered in vault");
  
  // Save deployment addresses
  const addresses = {
    safe: SAFE_ADDRESS,
    multisend: MULTISEND_ADDRESS,
    usdc: await usdc.getAddress(),
    usdt: await usdt.getAddress(),
    balancerVault: await vault.getAddress(),
    poolAddress: await bpt.getAddress(),
    poolId: poolId
  };
  
  console.log("\n=== Deployment Complete ===");
  console.log(JSON.stringify(addresses, null, 2));
  
  // Verify balances
  const usdcBalance = await usdc.balanceOf(SAFE_ADDRESS);
  const usdtBalance = await usdt.balanceOf(SAFE_ADDRESS);
  console.log("\nSafe Balances:");
  console.log("USDC:", hre.ethers.formatUnits(usdcBalance, 6));
  console.log("USDT:", hre.ethers.formatUnits(usdtBalance, 6));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
