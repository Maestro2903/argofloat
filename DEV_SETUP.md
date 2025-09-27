# Development Setup

## Quick Start

### Option 1: Run Both Services (Recommended)
```bash
# Install dependencies
npm install

# Run both frontend and backend
npm run dev:fullstack
# OR
./dev.sh
```

This will start:
- **Frontend**: http://localhost:5173
- **Backend**: http://localhost:8080

### Option 2: Run Services Separately

#### Frontend Only
```bash
npm run dev:frontend
# Runs on http://localhost:5173
```

#### Backend Only
```bash
npm run dev:backend
# Runs on http://localhost:8080
```

## Production Commands

### Build
```bash
npm run build
```

### Start Production (Both Services)
```bash
npm start
```

### Start Production Services Separately
```bash
# Frontend only
npm run start:frontend

# Backend only
npm run start:backend
```

## Environment Variables

### Development
- Frontend runs on port 5173
- Backend runs on port 8080

### Production
- `PORT` - Frontend port (default: 5173)
- `BACKEND_PORT` - Backend port (default: 8080)

## Requirements

- **Node.js** 18.13.0 - 22.x.x
- **Python** 3.11+
- **npm** or **yarn**

## Deployment

This project is configured for deployment on:
- **Vercel** (Frontend)
- **Render** (Full-stack with separate frontend/backend services)