#!/bin/bash

# Development script to run both frontend and backend
echo "🚀 Starting Float Chat development servers..."
echo ""
echo "Frontend will run on: http://localhost:5173"
echo "Backend will run on: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Run both services concurrently
npm run dev:fullstack