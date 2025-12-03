#!/usr/bin/env python
"""Minimal test FastAPI app"""

from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello"}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=7000, log_level='info')
