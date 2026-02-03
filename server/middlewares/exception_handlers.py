from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from logger import logger

async def catch_exceptions_middleware(request:Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "An internal server error occurred."},
        )