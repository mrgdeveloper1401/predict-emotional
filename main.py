import asyncio
import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI

load_dotenv()

# dotenv variable
debug = bool(os.getenv("DEBUG"))
port_number = int(os.getenv("PORT"))


app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port_number)