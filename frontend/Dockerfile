# =============================================================================
# frontend/Dockerfile — Debate-AI Next.js frontend (DEV configuration)
# =============================================================================
FROM node:20-alpine

WORKDIR /app

# ---- Dependencies (layer-cached) ----
# Copy package files FIRST so Docker caches the npm install layer.
# Code-only changes won't trigger a full reinstall.
COPY package.json package-lock.json* ./
RUN npm install

# ---- Application code ----
# Copied after npm install so code changes don't invalidate the dependency cache.
# In dev mode, overridden by bind mount in docker-compose.yml anyway.
COPY . .

EXPOSE 3000

# ---- Entrypoint ----
# Runs the Next.js dev server with hot reload.
# PRODUCTION: use "npm run build && npm start" or a multi-stage build.
CMD ["npm", "run", "dev"]
