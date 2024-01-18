#!/bin/bash

pip install ..

echo "Compiling C++ client..."
g++ -o client client.cpp -lrt -lpthread

echo "Running Python server..."
python server.py & # Running in background
SERVER_PID=$!
sleep 2 # Wait for the server to initialize

echo "Running C++ client..."
./client

kill $SERVER_PID # Terminate the Python server
