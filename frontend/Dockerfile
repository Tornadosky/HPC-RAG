FROM node:18-alpine AS build

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy project files
COPY . .

# Create a .env file at build time from environment variables
# This will be used during the build process
RUN echo "VITE_API_URL=${REACT_APP_API_URL:-http://localhost:8000}" > .env

# Build the app
RUN npm run build

# Runtime stage
FROM nginx:alpine

# Copy built assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html
COPY --from=build /app/public /usr/share/nginx/html/public

# Copy our custom Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Add runtime environment configuration script
WORKDIR /usr/share/nginx/html
COPY --from=build /app/package.json .

# Create a setup script to replace environment variables at runtime
RUN echo '#!/bin/sh' > /docker-entrypoint.d/00-update-env.sh && \
    echo 'envsubst < /usr/share/nginx/html/assets/index-*.js > /usr/share/nginx/html/assets/index-tmp.js && mv /usr/share/nginx/html/assets/index-tmp.js /usr/share/nginx/html/assets/index-*.js' >> /docker-entrypoint.d/00-update-env.sh && \
    chmod +x /docker-entrypoint.d/00-update-env.sh

# Expose port
EXPOSE 80

# Start nginx (uses default CMD from nginx image) 