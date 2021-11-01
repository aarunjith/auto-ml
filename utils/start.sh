#!/bin/sh
uvicorn api:app --host $APP_HOST --port $APP_PORT
