import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from solar_host import __version__
from solar_host.config import settings
from solar_host.models.base import BackendType
from solar_host.models_manager import ensure_models_dir, get_models_dir
from solar_host.process_manager import process_manager
from solar_host.routes import instances, jobs, models, websockets
from solar_host.ws_client import init_clients, get_clients, get_client, broadcast_health
from solar_host.jobs import JobExecutor, cleanup_loop, job_store
from solar_host.jobs.step_log_buffer import step_log_flush_loop

logger = logging.getLogger(__name__)


async def health_report_loop():
    """Periodically send health updates to all connected solar-controls."""
    while True:
        try:
            await asyncio.sleep(10)  # Report every 10 seconds
            clients = get_clients()
            if any(c.is_connected for c in clients):
                await broadcast_health()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("Health report error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the application"""
    key_hint = settings.api_key[:4] if len(settings.api_key) >= 4 else "***"
    logger.info("Starting Solar Host...")
    logger.info("API Key configured: %s...", key_hint)
    if settings.insecure:
        logger.warning(
            "INSECURE mode enabled - TLS certificate verification is disabled"
        )

    ensure_models_dir()
    logger.info("Models directory: %s", get_models_dir())

    # --- Job execution layer ---
    from solar_host.docker.service import DockerService
    from solar_host.docker.errors import DaemonUnavailableError

    docker_service: DockerService | None = None
    job_executor: JobExecutor | None = None
    cleanup_task: asyncio.Task | None = None
    try:
        docker_service = DockerService(settings)
        job_executor = JobExecutor(docker_service, job_store, settings)
        app.state.docker_service = docker_service
        app.state.job_executor = job_executor
        app.state.job_store = job_store
        cleanup_task = asyncio.create_task(cleanup_loop(job_store))
        logger.info("Docker service and job executor initialised")
    except DaemonUnavailableError:
        logger.warning(
            "Docker daemon unavailable — job executor disabled (training routes will return 503)"
        )
        app.state.docker_service = None
        app.state.job_executor = None
        app.state.job_store = job_store

    clients = init_clients(settings)
    health_task = None
    watchdog_task = None
    step_log_task: asyncio.Task | None = None
    if clients:
        for client in clients:
            await client.start()
        loop = asyncio.get_running_loop()
        process_manager.ensure_flush_loop(loop)
        step_log_task = asyncio.create_task(step_log_flush_loop())
        health_task = asyncio.create_task(health_report_loop())
        logger.info(
            "Solar Control WebSocket client(s) started (%d connection(s))", len(clients)
        )
    else:
        logger.info("Solar Control WebSocket client not configured (standalone mode)")

    watchdog_task = asyncio.create_task(process_manager.watchdog_loop())

    await process_manager.auto_restart_running_instances()
    logger.info("Solar Host started successfully")

    yield

    logger.info("Shutting down Solar Host...")

    # Stop active jobs and cancel the retention cleanup task.
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

    if job_executor is not None:
        for job in job_store.get_all():
            from solar_host.jobs.models import JobStatus as _JobStatus

            if job.status == _JobStatus.running:
                try:
                    await job_executor.cancel_job(job.job_id)
                except Exception:
                    logger.warning(
                        "Error cancelling job %r during shutdown", job.job_id
                    )
        await job_executor.await_all(timeout=30)

    if step_log_task:
        step_log_task.cancel()
        try:
            await step_log_task
        except asyncio.CancelledError:
            pass

    if health_task:
        health_task.cancel()
        try:
            await health_task
        except asyncio.CancelledError:
            pass

    if watchdog_task:
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

    for client in clients:
        await client.stop()


app = FastAPI(
    title="Solar Host",
    description="Process manager for model inference backends (llama.cpp, HuggingFace)",
    version=__version__,
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Key authentication middleware
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Verify API key for all requests except health check and OpenAPI docs"""
    # Allow CORS preflight requests (OPTIONS) without authentication
    if request.method == "OPTIONS":
        return await call_next(request)

    # Allow access to health check, docs, and OpenAPI schema
    public_paths = ["/health", "/", "/docs", "/redoc", "/openapi.json"]
    if request.url.path in public_paths:
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != settings.api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid or missing API key"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    return await call_next(request)


# Include routers
app.include_router(instances.router)
app.include_router(models.router)
app.include_router(websockets.router)
app.include_router(jobs.router)


# Customize OpenAPI schema to add security
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    }

    # Apply security to all paths except public ones
    public_paths = ["/health", "/", "/docs", "/redoc", "/openapi.json"]
    for path, path_item in openapi_schema["paths"].items():
        if path not in public_paths:
            for operation in path_item.values():
                if isinstance(operation, dict):
                    operation["security"] = [{"APIKeyHeader": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from solar_host.memory_monitor import get_disk_info

    disk = await asyncio.to_thread(get_disk_info, settings.models_dir)
    return {
        "status": "healthy",
        "service": "solar-host",
        "version": __version__,
        "disk": disk,
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "solar-host",
        "version": __version__,
        "description": "Process manager for model inference backends (llama.cpp, HuggingFace)",
        "supported_backends": [bt.value for bt in BackendType],
    }


@app.get("/memory")
async def get_memory():
    """Get GPU/RAM memory usage"""
    from fastapi import HTTPException
    from solar_host.memory_monitor import get_memory_info
    from solar_host.models import MemoryInfo

    memory_info = await asyncio.to_thread(get_memory_info)
    if not memory_info:
        raise HTTPException(status_code=503, detail="Memory information not available")
    try:
        return MemoryInfo(
            used_gb=float(memory_info["used_gb"]),
            total_gb=float(memory_info["total_gb"]),
            available_gb=float(memory_info["available_gb"]),
            percent=float(memory_info["percent"]),
            memory_type=str(memory_info["memory_type"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=503, detail=f"Memory data malformed: {e}")


@app.post("/reconnect")
async def reconnect_to_control():
    """Trigger reconnection to solar-control.

    Called by solar-control when it detects that this host's Socket.IO
    session has disconnected, prompting an immediate reconnection attempt
    instead of waiting for the next backoff cycle.
    """
    from fastapi import HTTPException

    client = get_client()
    if not client:
        raise HTTPException(
            status_code=503, detail="No solar-control client configured"
        )
    if client.is_connected:
        return {"status": "already_connected"}
    triggered = await client.reconnect()
    if triggered:
        return {"status": "reconnecting"}
    raise HTTPException(status_code=503, detail="Failed to trigger reconnection")


def run():
    """Entry point for the ``solar-host`` console script."""
    import uvicorn

    uvicorn.run("solar_host.main:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "solar_host.main:app", host=settings.host, port=settings.port, reload=True
    )
