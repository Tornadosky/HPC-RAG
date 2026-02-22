# Stage 1: Extract the static files from the original image
FROM hpcragregistry.azurecr.io/hpc-rag-frontend:latest as original
# We don't need to run anything here, just use it as a source for files

# Stage 2: Create a clean NGINX image with the files from stage 1
FROM nginx:stable-alpine

# Copy static files from the original image
COPY --from=original /usr/share/nginx/html /usr/share/nginx/html

# Copy our custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# In case the extraction failed, provide a fallback index.html
COPY index.html /usr/share/nginx/html/index.html.fallback

# Add a simple startup script to handle errors
RUN echo '#!/bin/sh\n\
# Check if index.html exists, if not use fallback\n\
if [ ! -f /usr/share/nginx/html/index.html ]; then\n\
  echo "Warning: index.html not found, using fallback"\n\
  cp /usr/share/nginx/html/index.html.fallback /usr/share/nginx/html/index.html\n\
fi\n\
# Start nginx\n\
exec nginx -g "daemon off;"\n\
' > /docker-entrypoint.sh && chmod +x /docker-entrypoint.sh

# Set environment variables that might be used by the app
ENV REACT_APP_API_URL=https://hpc-rag-backend.victoriousglacier-a42e5c9f.westeurope.azurecontainerapps.io
ENV API_URL=https://hpc-rag-backend.victoriousglacier-a42e5c9f.westeurope.azurecontainerapps.io
ENV NODE_ENV=production

# Expose port 80
EXPOSE 80

# Set the startup command to our simple script
CMD ["/docker-entrypoint.sh"]
