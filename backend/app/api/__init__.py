from .routes_health import router as health_router
from .routes_ingest import router as ingest_router
from .routes_search import router as search_router

__all__ = ["health_router", "ingest_router", "search_router"]
