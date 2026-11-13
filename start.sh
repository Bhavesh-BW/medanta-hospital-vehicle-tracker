#!/usr/bin/env bash
echo "Starting Medanta Hospital - Vehicle Access Control & ANPR System on http://localhost:8176..."
cd "$(dirname "$0")/backend"
python3 vehicle_tracker.py
