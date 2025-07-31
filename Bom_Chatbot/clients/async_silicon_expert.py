# clients/async_silicon_expert.py
"""Enhanced async Silicon Expert API client with performance optimizations."""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import List, Dict, Any

import aiohttp
import structlog

from Bom_Chatbot.config import SiliconExpertConfig
from Bom_Chatbot.constants import DEFAULT_PAGE_SIZE, MAX_SEARCH_RESULTS, HTTP_OK
from Bom_Chatbot.exceptions import (
    SiliconExpertError, AuthenticationError, ComponentSearchError, BOMError, ConfigurationError
)
from Bom_Chatbot.models import Component, SiliconExpertData, EnhancedComponent, BOMInfo
from Bom_Chatbot.services.progress import get_progress_tracker

logger = structlog.get_logger()


class CircuitBreaker:
    """Circuit breaker pattern for API resilience."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def can_proceed(self) -> bool:
        """Check if request can proceed."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False

        # HALF_OPEN state
        return True

    def on_success(self):
        """Handle successful request."""
        self.failure_count = 0
        self.state = "CLOSED"

    def on_failure(self):
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RateLimiter:
    """Adaptive rate limiter with backoff."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.current_delay = 1.0
        self.max_delay = 60.0

    async def acquire(self):
        """Acquire permission to make a request."""
        now = time.time()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests
                         if now - req_time < 60]

        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.requests.append(now)

    async def backoff_on_error(self):
        """Apply exponential backoff on error."""
        import random
        jitter = random.uniform(0.1, 0.3) * self.current_delay
        await asyncio.sleep(self.current_delay + jitter)
        self.current_delay = min(self.current_delay * 2, self.max_delay)

    def reset_backoff(self):
        """Reset backoff delay."""
        self.current_delay = 1.0


class ConnectionManager:
    """HTTP connection pool manager."""

    def __init__(self, timeout: int = 30):
        self.connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=10,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )

        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None

    @asynccontextmanager
    async def get_session(self):
        """Get or create HTTP session."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout,
                headers={'User-Agent': 'BOM-Agent/1.0'}
            )

        try:
            yield self.session
        except Exception:
            if self.session and not self.session.closed:
                await self.session.close()
            raise

    async def close(self):
        """Close connection manager."""
        if self.session and not self.session.closed:
            await self.session.close()
        await self.connector.close()


class AsyncSiliconExpertClient:
    """Enhanced async Silicon Expert API client."""

    def __init__(self, config: SiliconExpertConfig):
        self.config = config
        self.is_authenticated = False
        self.progress = get_progress_tracker()

        # Enhanced components
        self.connection_manager = ConnectionManager(timeout=config.timeout)
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)

        if not config.is_valid():
            raise ConfigurationError("Invalid Silicon Expert configuration")

    async def authenticate(self) -> bool:
        """Authenticate with Silicon Expert API."""
        if not self.circuit_breaker.can_proceed():
            raise SiliconExpertError("Circuit breaker is OPEN - too many failures")

        self.progress.info("Authentication", "Connecting to Silicon Expert API...")

        try:
            await self.rate_limiter.acquire()

            async with self.connection_manager.get_session() as session:
                async with session.post(
                        f"{self.config.base_url}/search/authenticateUser",
                        headers={'content-type': 'application/x-www-form-urlencoded'},
                        data={
                            'login': self.config.username,
                            'apiKey': self.config.api_key.get_secret_value()
                        }
                ) as response:

                    if response.status == HTTP_OK:
                        self.is_authenticated = True
                        self.circuit_breaker.on_success()
                        self.rate_limiter.reset_backoff()
                        self.progress.success("Authentication", "Successfully authenticated")
                        return True
                    else:
                        self.circuit_breaker.on_failure()
                        error_msg = f"HTTP {response.status}"
                        self.progress.error("Authentication", error_msg)
                        raise AuthenticationError(
                            f"Authentication failed: {error_msg}",
                            provider="Silicon Expert",
                            status_code=response.status
                        )

        except aiohttp.ClientError as e:
            self.circuit_breaker.on_failure()
            await self.rate_limiter.backoff_on_error()
            self.progress.error("Authentication", str(e))
            raise AuthenticationError(
                f"Authentication request failed: {str(e)}",
                provider="Silicon Expert"
            )

    async def _ensure_authenticated(self) -> None:
        """Ensure the client is authenticated."""
        if not self.is_authenticated:
            await self.authenticate()

    def _build_search_query(self, component: Component) -> str:
        """Build search query from component data."""
        search_parts = component.get_search_parts()
        return ' '.join(search_parts)

    async def search_component(self, component: Component, semaphore: asyncio.Semaphore) -> EnhancedComponent:
        """Search for a single component with concurrency control."""
        async with semaphore:  # Limit concurrent requests
            await self._ensure_authenticated()

            search_query = self._build_search_query(component)
            if not search_query:
                enhanced = EnhancedComponent(**asdict(component))
                enhanced.search_result = "No component data available for search"
                return enhanced

            if not self.circuit_breaker.can_proceed():
                enhanced = EnhancedComponent(**asdict(component))
                enhanced.search_result = "Circuit breaker OPEN - API temporarily unavailable"
                return enhanced

            try:
                await self.rate_limiter.acquire()

                # Log search attempt
                query_preview = search_query[:50] + "..." if len(search_query) > 50 else search_query
                logger.info("api_search_start", component=component.name, query=query_preview)

                params = {
                    'fmt': 'json',
                    'pageNumber': '1',
                    'pageSize': str(DEFAULT_PAGE_SIZE),
                    'description': search_query
                }

                async with self.connection_manager.get_session() as session:
                    async with session.get(
                            f"{self.config.base_url}/search/partsearch",
                            params=params
                    ) as response:
                        if response.status != HTTP_OK:
                            self.circuit_breaker.on_failure()
                            raise ComponentSearchError(
                                f"Search request failed with HTTP {response.status}",
                                component_name=component.name,
                                search_query=search_query,
                                status_code=response.status
                            )

                        api_data = await response.json()

                # Handle authentication errors
                if api_data and api_data.get('Status', {}).get('Code') == '39':
                    logger.info("re_authentication", reason="session_expired")
                    self.is_authenticated = False
                    await self._ensure_authenticated()
                    # Retry the request
                    return await self.search_component(component, semaphore)

                # Process successful results
                enhanced = EnhancedComponent(**asdict(component))

                if (api_data and
                        api_data.get('Status', {}).get('Success') == 'true' and
                        'Result' in api_data and
                        isinstance(api_data['Result'], list) and
                        len(api_data['Result']) > 0):

                    first_result = api_data['Result'][0]
                    match_rating = first_result.get('MatchRating', 'Unknown')
                    part_number_found = first_result.get('PartNumber', 'Unknown')

                    logger.info("component_found",
                                component=component.name,
                                part_number=part_number_found,
                                match_rating=match_rating)

                    # Create Silicon Expert data
                    se_data = SiliconExpertData(
                        com_id=first_result.get('ComID'),
                        part_number=first_result.get('PartNumber'),
                        manufacturer=first_result.get('Manufacturer'),
                        description=first_result.get('Description'),
                        lifecycle=first_result.get('Lifecycle'),
                        rohs=first_result.get('RoHS'),
                        rohs_version=first_result.get('RoHSVersion'),
                        datasheet=first_result.get('Datasheet'),
                        product_line=first_result.get('PlName'),
                        taxonomy_path=first_result.get('TaxonomyPath'),
                        match_rating=first_result.get('MatchRating'),
                        match_comment=first_result.get('MatchRatingComment'),
                        yeol=first_result.get('YEOL'),
                        resilience_rating=first_result.get('ResilienceRating'),
                        military_status=first_result.get('MilitaryStatus'),
                        aml_status=first_result.get('AMLStatus'),
                        search_query=search_query,
                        total_items=str(api_data.get('TotalItems', 'Unknown'))
                    )

                    # Add additional matches if available
                    if len(api_data['Result']) > 1:
                        for result in api_data['Result'][:MAX_SEARCH_RESULTS]:
                            match_info = {
                                'com_id': result.get('ComID'),
                                'part_number': result.get('PartNumber'),
                                'manufacturer': result.get('Manufacturer'),
                                'match_rating': result.get('MatchRating'),
                                'lifecycle': result.get('Lifecycle')
                            }
                            se_data.all_matches.append(match_info)

                    enhanced.silicon_expert_data = se_data
                    self.circuit_breaker.on_success()
                    self.rate_limiter.reset_backoff()

                else:
                    # Handle no results
                    error_msg = "No matching parts found"
                    if api_data and 'Status' in api_data:
                        status_msg = api_data['Status'].get('Message', '')
                        if status_msg and status_msg != "Successful Operation":
                            error_msg = f"API Message: {status_msg}"

                    enhanced.search_result = error_msg
                    logger.warning("no_component_match", component=component.name)

                return enhanced

            except ComponentSearchError:
                self.circuit_breaker.on_failure()
                await self.rate_limiter.backoff_on_error()
                raise
            except Exception as e:
                self.circuit_breaker.on_failure()
                await self.rate_limiter.backoff_on_error()
                logger.error("component_search_error", component=component.name, error=str(e))
                enhanced = EnhancedComponent(**asdict(component))
                enhanced.search_result = f"Search error: {str(e)}"
                return enhanced

    async def search_components_batch(self, components: List[Component],
                                      max_concurrent: int = 10) -> List[EnhancedComponent]:
        """Search for multiple components with controlled concurrency."""
        if not components:
            return []

        self.progress.info("Batch Search",
                           f"Processing {len(components)} components with max {max_concurrent} concurrent requests...")

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        # Create tasks for all components
        tasks = [
            self.search_component(component, semaphore)
            for component in components
        ]

        try:
            # Execute all searches concurrently
            enhanced_components = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle exceptions
            results = []
            successful_searches = 0
            failed_searches = 0

            for i, result in enumerate(enhanced_components):
                if isinstance(result, Exception):
                    # Handle exception case
                    logger.error("batch_search_exception",
                                 component_index=i,
                                 component_name=components[i].name if i < len(components) else "unknown",
                                 error=str(result))
                    enhanced = EnhancedComponent(**asdict(components[i]))
                    enhanced.search_result = f"Search failed: {str(result)}"
                    results.append(enhanced)
                    failed_searches += 1
                else:
                    # Handle successful result
                    if result.silicon_expert_data:
                        successful_searches += 1
                    else:
                        failed_searches += 1
                    results.append(result)

            self.progress.success(
                "Batch Search",
                f"Completed {len(components)} components - {successful_searches} successful, {failed_searches} failed"
            )

            return results

        except Exception as e:
            self.progress.error("Batch Search", f"Batch search failed: {str(e)}")
            # Return failed components
            return [
                EnhancedComponent(**asdict(comp), search_result=f"Batch search failed: {str(e)}")
                for comp in components
            ]

    async def close(self):
        """Clean up resources."""
        await self.connection_manager.close()

    # Sync wrapper methods for backward compatibility
    def search_components(self, components: List[Component]) -> List[EnhancedComponent]:
        """Synchronous wrapper for batch search."""
        return asyncio.run(self.search_components_batch(components))

    # Other methods remain the same but should be converted to async
    # For brevity, I'll show the pattern for one more method

    async def create_empty_bom_async(self, bom_info: BOMInfo) -> Dict[str, Any]:
        """Create an empty BOM asynchronously."""
        await self._ensure_authenticated()

        payload = {
            "name": bom_info.name,
            "columns": bom_info.columns,
            "description": bom_info.description
        }

        if bom_info.parent_path:
            payload["parentPath"] = bom_info.parent_path

        try:
            await self.rate_limiter.acquire()

            async with self.connection_manager.get_session() as session:
                async with session.post(
                        f"{self.config.base_url}/bom/add-empty-bom",
                        headers={'Content-Type': 'application/json'},
                        json=payload
                ) as response:

                    if response.status == HTTP_OK:
                        return await response.json()
                    else:
                        raise BOMError(
                            f"Failed to create BOM: HTTP {response.status}",
                            bom_name=bom_info.name,
                            status_code=response.status,
                            response_data=await response.text()
                        )

        except aiohttp.ClientError as e:
            raise BOMError(
                f"BOM creation request failed: {str(e)}",
                bom_name=bom_info.name
            )

    # Sync wrapper for backward compatibility
    def create_empty_bom(self, bom_info: BOMInfo) -> Dict[str, Any]:
        """Synchronous wrapper for BOM creation."""
        return asyncio.run(self.create_empty_bom_async(bom_info))