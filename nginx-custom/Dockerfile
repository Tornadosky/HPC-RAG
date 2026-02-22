FROM nginx:stable-alpine

# Remove default nginx website
RUN rm -rf /usr/share/nginx/html/*

# Copy custom configuration and content
COPY default.conf /etc/nginx/conf.d/default.conf
COPY index.html /usr/share/nginx/html/

# Make sure nginx can access all files
RUN chmod -R 755 /usr/share/nginx/html

EXPOSE 80

# Start nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
