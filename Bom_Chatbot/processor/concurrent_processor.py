# async/concurrent_processor.py
"""Enhanced async processing for concurrent operations."""

import asyncio
import time
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from Bom_Chatbot.models import Component, EnhancedComponent
from Bom_Chatbot.services.progress import get_progress_tracker


@dataclass
class ProcessingConfig:
    """Configuration for concurrent processing."""
    max_concurrent: int = 10
    batch_size: int = 50
    timeout_per_item: float = 30.0
    enable_batching: bool = True
    thread_pool_size: int = 4


class ConcurrentProcessor:
    """Enhanced concurrent processing with adaptive batching."""

    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.progress = get_progress_tracker()
        self.executor = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        self.performance_stats = {
            "total_processed": 0,
            "avg_processing_time": 0.0,
            "success_rate": 0.0
        }

    async def process_components_concurrent(
            self,
            components: List[Component],
            processor_func: Callable,
            progress_callback: Optional[Callable] = None
    ) -> List[EnhancedComponent]:
        """Process components concurrently with adaptive batching."""

        if not components:
            return []

        start_time = time.time()
        total_components = len(components)

        self.progress.info("Concurrent Processing",
                           f"Processing {total_components} components with {self.config.max_concurrent} workers")

        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # Determine optimal batch size
        batch_size = self._calculate_optimal_batch_size(total_components)

        results = []
        successful = 0
        failed = 0

        # Process in batches if enabled and beneficial
        if self.config.enable_batching and total_components > batch_size:
            results = await self._process_in_batches(
                components, processor_func, semaphore, batch_size, progress_callback
            )
        else:
            # Process all at once
            results = await self._process_concurrent_batch(
                components, processor_func, semaphore, progress_callback
            )

        # Calculate statistics
        for result in results:
            if isinstance(result, EnhancedComponent) and result.silicon_expert_data:
                successful += 1
            else:
                failed += 1

        processing_time = time.time() - start_time

        # Update performance stats
        self._update_performance_stats(total_components, processing_time, successful)

        self.progress.success("Concurrent Processing",
                              f"Completed {total_components} components in {processing_time:.1f}s "
                              f"({successful} successful, {failed} failed)")

        return results

    async def _process_in_batches(
            self,
            components: List[Component],
            processor_func: Callable,
            semaphore: asyncio.Semaphore,
            batch_size: int,
            progress_callback: Optional[Callable]
    ) -> List[EnhancedComponent]:
        """Process components in batches."""

        all_results = []
        total_batches = (len(components) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(components), batch_size):
            batch = components[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1

            self.progress.info("Batch Processing",
                               f"Processing batch {batch_num}/{total_batches} ({len(batch)} components)")

            batch_results = await self._process_concurrent_batch(
                batch, processor_func, semaphore, progress_callback
            )

            all_results.extend(batch_results)

            # Optional delay between batches to respect rate limits
            if batch_num < total_batches:
                await asyncio.sleep(0.1)

        return all_results

    async def _process_concurrent_batch(
            self,
            components: List[Component],
            processor_func: Callable,
            semaphore: asyncio.Semaphore,
            progress_callback: Optional[Callable]
    ) -> List[EnhancedComponent]:
        """Process a single batch concurrently."""

        async def process_with_semaphore(component: Component, index: int) -> EnhancedComponent:
            async with semaphore:
                try:
                    if asyncio.iscoroutinefunction(processor_func):
                        result = await processor_func(component, semaphore)
                    else:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            self.executor, processor_func, component
                        )

                    if progress_callback:
                        await progress_callback(index + 1, len(components), component.name)

                    return result

                except Exception as e:
                    self.progress.error("Component Processing", f"Failed {component.name}: {e}")
                    # Return failed component
                    from dataclasses import asdict
                    enhanced = EnhancedComponent(**asdict(component))
                    enhanced.search_result = f"Processing failed: {str(e)}"
                    return enhanced

        # Create tasks for all components
        tasks = [
            process_with_semaphore(component, idx)
            for idx, component in enumerate(components)
        ]

        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.progress.error("Batch Processing", f"Task {i} failed: {result}")
                    # Create failed component
                    from dataclasses import asdict
                    enhanced = EnhancedComponent(**asdict(components[i]))
                    enhanced.search_result = f"Task failed: {str(result)}"
                    processed_results.append(enhanced)
                else:
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            self.progress.error("Batch Processing", f"Batch failed: {e}")
            # Return all as failed
            from dataclasses import asdict
            return [
                EnhancedComponent(**asdict(comp), search_result=f"Batch failed: {str(e)}")
                for comp in components
            ]

    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on total items and performance."""
        base_batch_size = self.config.batch_size

        # Adjust based on total items
        if total_items < 20:
            return total_items  # Process all at once for small sets
        elif total_items < 100:
            return min(25, total_items)
        elif total_items < 500:
            return min(50, total_items)
        else:
            return min(100, total_items)

    def _update_performance_stats(self, processed: int, time_taken: float, successful: int):
        """Update performance statistics."""
        self.performance_stats["total_processed"] += processed

        # Update running average of processing time
        current_avg = self.performance_stats["avg_processing_time"]
        total_previous = self.performance_stats["total_processed"] - processed

        if total_previous > 0:
            self.performance_stats["avg_processing_time"] = (
                    (current_avg * total_previous + time_taken) / self.performance_stats["total_processed"]
            )
        else:
            self.performance_stats["avg_processing_time"] = time_taken

        # Update success rate
        self.performance_stats["success_rate"] = (successful / processed) * 100

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics report."""
        return {
            "total_processed": self.performance_stats["total_processed"],
            "average_processing_time": round(self.performance_stats["avg_processing_time"], 2),
            "success_rate": round(self.performance_stats["success_rate"], 1),
            "configuration": {
                "max_concurrent": self.config.max_concurrent,
                "batch_size": self.config.batch_size,
                "timeout_per_item": self.config.timeout_per_item
            }
        }

    async def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)