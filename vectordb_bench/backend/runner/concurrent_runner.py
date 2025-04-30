# vectordb_bench/backend/runner/concurrent_runner.py

import threading
import queue
import time
import logging
from typing import Generator, Optional, List
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from vectordb_bench.backend.cases import ConcurrentReadWriteConfig
from vectordb_bench.backend.clients import api
from vectordb_bench.backend.dataset import DatasetManager
from vectordb_bench.metric import WindowedMetricsCollector, WindowMetrics, calc_recall

log = logging.getLogger(__name__)



class ConcurrentReadWriteRunner:
    """Runner for concurrent insert and search operations"""
    
    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        config: ConcurrentReadWriteConfig,
        metrics_collector: WindowedMetricsCollector,
        ground_truth: Optional[pd.DataFrame] = None
    ):
        self.db = db
        self.dataset = dataset
        self.metrics = metrics_collector
        self.config = config
        self.ground_truth = ground_truth
        
        # Enhanced batch processing
        self.batch_size = config.batch_size
        self.insert_queue = queue.Queue(maxsize=1000)
        self.search_queue = queue.Queue(maxsize=1000)
        


    def _process_batch(self, embeddings: List[List[float]], metadata: List[int]) -> None:
        """Process a batch of insertions with metrics collection"""
        start_time = time.time()
        count, error = self.db.insert_embeddings(embeddings, metadata)
        latency = time.time() - start_time
        
        if error:
            raise RuntimeError(f"Insert error: {error}")
            
        self.metrics.record_batch(len(embeddings), latency)
        self.metrics.record_insert(count, latency)
        
    def _process_search(self, query: List[float], query_id: int) -> None:
        """Process a search query with metrics collection"""
        start_time = time.time()
        results = self.db.search_embedding(query, k=self.config.search_k)
        latency = time.time() - start_time
        
        recall = 0.0
        if self.ground_truth is not None:
            gt = self.ground_truth.iloc[query_id]["neighbors_id"]
            recall = calc_recall(self.config.search_k, gt, results)
            
        self.metrics.record_search(latency, recall)

        
    def _insert_worker(self):
        """Worker thread for handling insertions"""
        try:
            with self.db.init():
                while not self.stop_event.is_set():
                    try:
                        batch = self.insert_queue.get(timeout=1)
                        if batch is None:
                            break
                            
                        embeddings, metadata = batch
                        start_time = time.time()
                        count, error = self.db.insert_embeddings(embeddings, metadata)
                        latency = time.time() - start_time
                        
                        if error:
                            log.error(f"Insert error: {error}")
                            self.error_event.set()
                            break
                            
                        self.metrics.record_insert(count, latency)
                        
                    except queue.Empty:
                        continue
        except Exception as e:
            log.error(f"Insert worker error: {e}")
            self.error_event.set()
            
    def _search_worker(self):
        """Worker thread for handling searches"""
        try:
            with self.db.init():
                while not self.stop_event.is_set():
                    try:
                        query = self.search_queue.get(timeout=1)
                        if query is None:
                            break
                            
                        start_time = time.time()
                        results = self.db.search_embedding(
                            query, 
                            k=self.config.search_k
                        )
                        latency = time.time() - start_time
                        
                        # Calculate recall if ground truth available
                        recall = 0
                        if self.ground_truth is not None:
                            recall = calc_recall(
                                self.config.search_k,
                                self.ground_truth[query_id],
                                results
                            )
                        
                        self.metrics.record_search(latency, recall)
                        
                    except queue.Empty:
                        continue
        except Exception as e:
            log.error(f"Search worker error: {e}")
            self.error_event.set()
            
    def run(self) -> Generator[WindowMetrics, None, None]:
        """Run concurrent operations and yield metrics windows"""
        
        # Start worker pools
        with ThreadPoolExecutor(max_workers=self.config.num_insert_workers) as insert_pool, \
             ThreadPoolExecutor(max_workers=self.config.num_search_workers) as search_pool:
            
            # Submit worker tasks
            insert_futures = [
                insert_pool.submit(self._insert_worker)
                for _ in range(self.config.num_insert_workers)
            ]
            search_futures = [
                search_pool.submit(self._search_worker)
                for _ in range(self.config.num_search_workers)
            ]
            
            start_time = time.time()
            try:
                while time.time() - start_time < self.config.total_duration:
                    if self.error_event.is_set():
                        raise RuntimeError("Worker error occurred")
                        
                    # Feed insert queue
                    if not self.insert_queue.full():
                        batch = next(self.dataset)
                        self.insert_queue.put((batch['emb'], batch['id']))
                    
                    # Feed search queue
                    if not self.search_queue.full():
                        query = self._get_random_query()
                        self.search_queue.put(query)
                    
                    # Yield completed windows
                    if self.metrics.windows:
                        yield self.metrics.windows[-1]
                    
                    time.sleep(0.1)
                    
            finally:
                # Cleanup
                self.stop_event.set()
                
                # Signal workers to stop
                for _ in range(self.config.num_insert_workers):
                    self.insert_queue.put(None)
                for _ in range(self.config.num_search_workers):
                    self.search_queue.put(None)
                    
                # Wait for workers to finish
                for f in insert_futures + search_futures:
                    f.result()