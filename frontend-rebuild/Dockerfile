FROM nginx:stable-alpine

# Remove default nginx website
RUN rm -rf /usr/share/nginx/html/*

# First, copy all frontend files
COPY extract /usr/share/nginx/html/

# Copy our custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Test NGINX config at build time to catch any errors
RUN nginx -t

# Set environment variables
ENV REACT_APP_API_URL=https://hpc-rag-backend.victoriousglacier-a42e5c9f.westeurope.azurecontainerapps.io
ENV API_URL=https://hpc-rag-backend.victoriousglacier-a42e5c9f.westeurope.azurecontainerapps.io
ENV NODE_ENV=production

# Fix permissions
RUN chmod -R 755 /usr/share/nginx/html

# Start NGINX directly - no custom entrypoint
CMD ["nginx", "-g", "daemon off;"]
