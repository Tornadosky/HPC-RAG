import axios from 'axios';

// Create a custom instance of axios with default settings
const api = axios.create({
  baseURL: '',  // Use empty baseURL to work with relative URLs
  timeout: 30000,  // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
});

export default api; 