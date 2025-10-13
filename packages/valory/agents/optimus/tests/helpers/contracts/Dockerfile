FROM node:18-alpine

WORKDIR /app

# Copy contract files first
COPY package.json hardhat.config.js ./
COPY mocks/ ./contracts/
COPY deploy_contracts.js ./

# Install dependencies
RUN npm install

# Create scripts directory and move deploy script
RUN mkdir -p scripts && mv deploy_contracts.js scripts/

# Compile contracts
RUN npx hardhat compile

# Expose Hardhat port
EXPOSE 8545

# Run Hardhat node and deploy contracts
CMD ["sh", "-c", "npx hardhat node --hostname 0.0.0.0 & sleep 10 && npx hardhat run scripts/deploy_contracts.js --network localhost && tail -f /dev/null"]
