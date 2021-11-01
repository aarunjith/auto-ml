#!/bin/sh
cd api
uvicorn api:app --host $APP_HOST --port $APP_PORT
