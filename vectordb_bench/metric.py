import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
import numpy as np
import psutil
import os

log = logging.getLogger(__name__)


# vectordb_bench/metric.py
# For windowed incremental performance measurement


@dataclass
class HardwareMetrics:
    """Hardware-level performance metrics"""
    l1_cache_misses: int = 0
    l3_cache_misses: int = 0
    peak_memory_kb: int = 0
    cpu_usage_percent: float = 0.0
    io_counters: Dict = None


@dataclass
class WindowMetrics:
    """Enhanced metrics collected within a single time window"""
    window_id: int
    start_time: float
    end_time: float

    # Insert metrics
    insert_count: int
    insert_latencies: List[float]
    insert_throughput: float = 0.0

    # Search metrics
    search_count: int
    search_latencies: List[float]
    search_recalls: List[float]
    search_qps: float = 0.0

    # Hardware metrics
    hardware_metrics: Optional[HardwareMetrics] = None

    # Batch metrics
    batch_size: int = 0
    batch_latencies: List[float] = None

    @property
    def window_duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def avg_insert_latency(self) -> float:
        return np.mean(self.insert_latencies) if self.insert_latencies else 0

    @property
    def p95_insert_latency(self) -> float:
        return np.percentile(self.insert_latencies, 95) if self.insert_latencies else 0

    @property
    def p99_insert_latency(self) -> float:
        return np.percentile(self.insert_latencies, 99) if self.insert_latencies else 0

    @property
    def avg_search_latency(self) -> float:
        return np.mean(self.search_latencies) if self.search_latencies else 0

    @property
    def p95_search_latency(self) -> float:
        return np.percentile(self.search_latencies, 95) if self.search_latencies else 0

    @property
    def p99_search_latency(self) -> float:
        return np.percentile(self.search_latencies, 99) if self.search_latencies else 0

    @property
    def avg_recall(self) -> float:
        return np.mean(self.search_recalls) if self.search_recalls else 0


class HardwareMetricsCollector:
    """Collector for hardware-level metrics"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def collect_metrics(self) -> HardwareMetrics:
        metrics = HardwareMetrics()
        metrics.peak_memory_kb = self.process.memory_info().rss / 1024
        metrics.cpu_usage_percent = self.process.cpu_percent()
        metrics.io_counters = self.process.io_counters()._asdict()
        return metrics


class WindowedMetricsCollector:
    """Enhanced thread-safe collector for operation metrics"""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.hardware_collector = HardwareMetricsCollector()
        self._reset_window()
        self.windows: List[WindowMetrics] = []
        self.window_id = 0

    def _reset_window(self):
        """Initialize a new metrics window"""
        self.current_window = {
            'start_time': time.time(),
            'insert_count': 0,
            'insert_latencies': [],
            'search_count': 0,
            'search_latencies': [],
            'search_recalls': [],
            'batch_latencies': [],
            'batch_size': 0
        }

    def record_batch(self, size: int, latency: float):
        """Record metrics for a batch operation"""
        self.current_window['batch_size'] = size
        self.current_window['batch_latencies'].append(latency)
        self._check_window()

    def record_insert(self, count: int, latency: float):
        """Record metrics for an insert operation"""
        self.current_window['insert_count'] += count
        self.current_window['insert_latencies'].append(latency)
        self._check_window()

    def record_search(self, latency: float, recall: float):
        """Record metrics for a search operation"""
        self.current_window['search_count'] += 1
        self.current_window['search_latencies'].append(latency)
        self.current_window['search_recalls'].append(recall)
        self._check_window()

    def _check_window(self):
        """Check if current window should be closed"""
        total_ops = (self.current_window['insert_count'] +
                     self.current_window['search_count'])
        if total_ops >= self.window_size:
            self._close_window()

    def _close_window(self):
        """Close current window and create metrics"""
        end_time = time.time()
        window_duration = end_time - self.current_window['start_time']

        # Collect hardware metrics
        hardware_metrics = self.hardware_collector.collect_metrics()

        metrics = WindowMetrics(
            window_id=self.window_id,
            start_time=self.current_window['start_time'],
            end_time=end_time,
            insert_count=self.current_window['insert_count'],
            insert_latencies=self.current_window['insert_latencies'],
            insert_throughput=self.current_window['insert_count'] / window_duration,
            search_count=self.current_window['search_count'],
            search_latencies=self.current_window['search_latencies'],
            search_recalls=self.current_window['search_recalls'],
            search_qps=self.current_window['search_count'] / window_duration,
            hardware_metrics=hardware_metrics,
            batch_size=self.current_window['batch_size'],
            batch_latencies=self.current_window['batch_latencies']
        )

        self.windows.append(metrics)
        self.window_id += 1
        self._reset_window()


@dataclass
class Metric:
    """result metrics"""

    # for load cases
    max_load_count: int = 0

    # for performance cases
    load_duration: float = 0.0  # duration to load all dataset into DB
    qps: float = 0.0
    serial_latency_p99: float = 0.0
    recall: float = 0.0
    ndcg: float = 0.0
    conc_num_list: list[int] = field(default_factory=list)
    conc_qps_list: list[float] = field(default_factory=list)
    conc_latency_p99_list: list[float] = field(default_factory=list)
    conc_latency_avg_list: list[float] = field(default_factory=list)


QURIES_PER_DOLLAR_METRIC = "QP$ (Quries per Dollar)"
LOAD_DURATION_METRIC = "load_duration"
SERIAL_LATENCY_P99_METRIC = "serial_latency_p99"
MAX_LOAD_COUNT_METRIC = "max_load_count"
QPS_METRIC = "qps"
RECALL_METRIC = "recall"

metric_unit_map = {
    LOAD_DURATION_METRIC: "s",
    SERIAL_LATENCY_P99_METRIC: "ms",
    MAX_LOAD_COUNT_METRIC: "K",
    QURIES_PER_DOLLAR_METRIC: "K",
}

lower_is_better_metrics = [
    LOAD_DURATION_METRIC,
    SERIAL_LATENCY_P99_METRIC,
]

metric_order = [
    QPS_METRIC,
    RECALL_METRIC,
    LOAD_DURATION_METRIC,
    SERIAL_LATENCY_P99_METRIC,
    MAX_LOAD_COUNT_METRIC,
]


def isLowerIsBetterMetric(metric: str) -> bool:
    return metric in lower_is_better_metrics


def calc_recall(count: int, ground_truth: list[int], got: list[int]) -> float:
    recalls = np.zeros(count)
    for i, result in enumerate(got):
        if result in ground_truth:
            recalls[i] = 1

    return np.mean(recalls)


def get_ideal_dcg(k: int):
    ideal_dcg = 0
    for i in range(k):
        ideal_dcg += 1 / np.log2(i + 2)

    return ideal_dcg


def calc_ndcg(ground_truth: list[int], got: list[int], ideal_dcg: float) -> float:
    dcg = 0
    ground_truth = list(ground_truth)
    for got_id in set(got):
        if got_id in ground_truth:
            idx = ground_truth.index(got_id)
            dcg += 1 / np.log2(idx + 2)
    return dcg / ideal_dcg
