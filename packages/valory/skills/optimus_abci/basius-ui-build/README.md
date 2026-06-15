# BabyDegen UI

React application for BabyDegen UI.
Served by the Modius, Optimus and Basius agents, designed to be consumed by the agent and available in [Pearl](https://github.com/valory-xyz/olas-operate-app).

## 🚀 Development

1. Install via `yarn install`
2. Run via `npx nx serve babydegen-ui`
    - The app will be available at `http://localhost:4300`
    - For modius, update the REACT_APP_AGENT_NAME value in .env file to `modius`
    - For optimus, update the REACT_APP_AGENT_NAME value in .env file to `optimus`
    - For basius, update the REACT_APP_AGENT_NAME value in .env file to `basius`
3. Build for production via `npx nx build babydegen-ui`
    - The build will be available in the `dist/apps/babydegen-ui` directory
    - `/build` is the output directory, and can be served statically

## 🧪 Mock Data
To mock, set `IS_MOCK_ENABLED=true` in `.env` and the app will use mock data instead of the API. The chat mock is gated by `isChatEnabled` in `mockFeatures.ts`.

## 📦 Release process

1. Bump the version in `package.json`
2. Push a new tag to the repository
    - For modius, use suffix `-modius` (e.g., `v1.0.0-modius`)
    - For optimus, use suffix `-optimus` (e.g., `v1.0.0-optimus`)
    - For basius, use suffix `-basius` (e.g., `v1.0.0-basius`)
3. The CI will build and release the contents of the `dist/apps/babydegen-ui` directory to a zip file.

## 🔐 Deployment expectations

This app ships as a static-asset ZIP attached to a GitHub Release. The downstream operator (typically the Pearl agent container) is responsible for runtime security headers. Recommended minimum set when serving the unpacked bundle:

| Header | Recommended value | Why |
| --- | --- | --- |
| `Content-Security-Policy` | `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; img-src 'self' data:; font-src 'self' data: https://fonts.gstatic.com; connect-src 'self' http://127.0.0.1:8716; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'` | `'unsafe-inline'` for `style-src` is required by styled-components 5.x. Google Fonts CSS + font files need explicit entries. Backend fetches go to `http://127.0.0.1:8716` via `LOCAL` (see [`libs/util-constants-and-types`](../../libs/util-constants-and-types/src/lib/constants/local.ts)). `frame-ancestors 'none'` blocks clickjacking and **only works as a header** (meta-tag is ignored). |
| `Strict-Transport-Security` | `max-age=31536000` | If served over HTTPS. |
| `X-Content-Type-Options` | `nosniff` | Disable MIME-type sniffing. |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Limit referrer leakage. |
| `Permissions-Policy` | `camera=(), microphone=(), geolocation=(), payment=()` | Disable browser features the app does not use. |

If the operator cannot set HTTP headers, an equivalent (weaker — no `frame-ancestors`) `<meta http-equiv="Content-Security-Policy">` can be injected into `index.html` post-build. See [`SUPPLY-CHAIN-SECURITY.md`](../../SUPPLY-CHAIN-SECURITY.md) for the threat model that drives these recommendations.
