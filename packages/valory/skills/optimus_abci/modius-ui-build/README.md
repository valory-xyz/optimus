# BabyDegen UI

React application for BabyDegen UI.
Served by the Modius and Optimus agent, designed to be consumed by the agent and available in [Pearl]

## 🚀 Development

1. Install via `yarn install`
2. Run via `npx nx serve babydegen-ui`
    - The app will be available at `http://localhost:4200`
    - For modius, update the REACT_APP_AGENT_NAME value in .env file to `modius`
    - For optimus, update the REACT_APP_AGENT_NAME value in .env file to `optimus` 
3. Build for production via `npx nx build babydegen-ui`
    - The build will be available in the `dist/apps/babydegen-ui` directory
    - `/build` is the output directory, and can be served statically

## 🧪 Mock Data
To mock, update the `IS_MOCK_ENABLED` in `config.ts` to `true` and the app will use the mock data instead of the API. To enable the chat mock, set `isChatEnabled` in `mockFeatures.ts` to `true` as well.

## 📦 Release process

1. Bump the version in `package.json`
2. Push a new tag to the repository
    - For modius, use suffix `-modius` (e.g., `v1.0.0-modius`)
    - For optimus, use suffix `-optimus` (e.g., `v1.0.0-optimus`)
3. The CI will build and release the contents of the `dist/apps/babydegen-ui` directory to a zip file.
