// Custom API configuration with direct backend URL
(function() {
  console.log('Initializing custom API handler');
  
  // We need to wait a bit for the main JS to load
  setTimeout(function() {
    // Function to extract and clean the API path
    function getCleanPath(url) {
      // If URL starts with /api/, remove that prefix
      if (url && typeof url === 'string' && url.startsWith('/api/')) {
        return url.substring(5); // Remove /api/ prefix
      }
      return url;
    }
    
    // Simple fetch-based API implementation as fallback
    const directApi = {
      post: async function(path, data) {
        const cleanPath = getCleanPath(path);
        const url = 'https://hpc-rag-backend.victoriousglacier-a42e5c9f.westeurope.azurecontainerapps.io/' + cleanPath;
        
        console.log('Direct POST request to:', url);
        
        try {
          const response = await fetch(url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
          });
          
          if (!response.ok) {
            const errMsg = 'API error: ' + response.status;
            console.error(errMsg);
            throw new Error(errMsg);
          }
          
          const result = await response.json();
          return { data: result };
        } catch (error) {
          console.error('API request failed:', error);
          throw error;
        }
      }
    };
    
    // First check for window.axios
    if (window.axios) {
      console.log('Found global axios, intercepting requests');
      const originalAxios = window.axios;
      
      // Add request interceptor to modify API URLs
      originalAxios.interceptors.request.use(
        function(config) {
          if (config.url && typeof config.url === 'string' && config.url.startsWith('/api/')) {
            const originalUrl = config.url;
            // Extract path without /api/ prefix
            const path = getCleanPath(config.url);
            // Set the full URL with backend address
            config.url = 'https://hpc-rag-backend.victoriousglacier-a42e5c9f.westeurope.azurecontainerapps.io/' + path;
            console.log('Modified axios URL:', originalUrl, '->', config.url);
          }
          return config;
        },
        function(error) {
          console.error('Axios request error:', error);
          return Promise.reject(error);
        }
      );
    } 
    else {
      console.log('No axios found, using direct API');
      // Use our fallback API implementation
      window.api = directApi;
      
      // Try to intercept XMLHttpRequest for safety
      try {
        const originalOpen = XMLHttpRequest.prototype.open;
        XMLHttpRequest.prototype.open = function(method, url, ...args) {
          let newUrl = url;
          if (typeof url === 'string' && url.includes('/api/')) {
            newUrl = url.replace('/api/', 'https://hpc-rag-backend.victoriousglacier-a42e5c9f.westeurope.azurecontainerapps.io/');
            console.log('XHR intercepted:', url, '->', newUrl);
          }
          return originalOpen.call(this, method, newUrl, ...args);
        };
      } catch (e) {
        console.error('Failed to override XMLHttpRequest:', e);
      }
    }
  }, 500); // Give it some time to load
})();
