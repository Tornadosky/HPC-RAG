# HPC Framework Recommender

A web application that recommends the best parallel programming framework for HPC engineers based on their hardware, priorities, and project requirements.

## Features

- Wizard-style survey with 3 easy steps
- Interactive UI with dark/light mode
- ML-powered recommendations with probability scores
- Beautiful visualizations

## Tech Stack

- **Frontend**: React, Vite, Chakra UI, Chart.js
- **Backend**: FastAPI, scikit-learn
- **ML Model**: Calibrated Random Forest
- **Deployment**: Docker Compose

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Node.js 16+ (for local development)
- Python 3.9+ (for local development)

### Running with Docker

The easiest way to run the application is with Docker Compose:

```bash
# Build and start containers
docker-compose up --build

# The app will be available at http://localhost:3000
# The API will be available at http://localhost:8000
```

### Local Development

#### Backend (FastAPI)

```bash
# Install dependencies
cd api
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload
```

#### Frontend (React)

```bash
# Install dependencies
cd frontend
npm install

# Start development server
npm run dev
```

## Usage

1. Start the application
2. Navigate to http://localhost:3000
3. Click "Start the Survey"
4. Answer questions about your hardware, priorities, and project
5. View your personalized framework recommendations

## API Documentation

When running the backend, access the Swagger documentation at:

```
http://localhost:8000/docs
```

This provides an interactive API documentation with all available endpoints.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
