"""Bounded background execution with plain results and UI-thread publication."""

import heapq
import itertools
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import SimpleQueue
from threading import Condition, Event, Lock, Thread
from uuid import uuid4


class Cancellation:
    def __init__(self):
        self.event = Event()

    def cancel(self):
        self.event.set()

    def check(self):
        if self.event.is_set():
            raise CancelledError()


@dataclass(frozen=True)
class Job:
    document_id: str
    generation: int
    state_id: str
    operation_id: str = field(default_factory=lambda: uuid4().hex)
    cancellation: Cancellation = field(default_factory=Cancellation)
    preview_key: str | None = None
    publication: str = "current"
    priority: int = 10
    resource_key: str | None = None


class PriorityExecutor:
    """Small non-preemptive priority pool; running work remains cooperative."""

    def __init__(self, workers):
        self.condition = Condition()
        self.queue = []
        self.sequence = itertools.count()
        self.closed = False
        self.threads = [
            Thread(target=self._run, name=f"pypaint-{i}", daemon=False) for i in range(workers)
        ]
        for thread in self.threads:
            thread.start()

    def submit(self, function, priority=10):
        future = Future()
        with self.condition:
            if self.closed:
                raise RuntimeError("Executor is closed")
            heapq.heappush(self.queue, (priority, next(self.sequence), future, function))
            self.condition.notify()
        return future

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.queue or self.closed)
                if not self.queue:
                    return
                _, _, future, function = heapq.heappop(self.queue)
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(function())
                except BaseException as error:
                    future.set_exception(error)

    def shutdown(self, wait=True, cancel_futures=False):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if wait:
            for thread in self.threads:
                thread.join()


@dataclass(frozen=True)
class Result:
    job: Job
    value: object = None
    error: Exception | None = None


class JobQueue:
    def __init__(self, workers=2, max_tasks=8, max_bytes=128 * 1024**2):
        self.executor = PriorityExecutor(workers)
        self.max_tasks, self.max_bytes = max_tasks, max_bytes
        self.inflight_bytes = 0
        self.pending = {}
        self.previews = {}
        self.results = SimpleQueue()
        self.lock = Lock()
        self.closed = False
        self.discarded = []

    def submit(self, job, function, reserved_bytes):
        with self.lock:
            if job.publication not in ("current", "saved") or job.operation_id in self.pending:
                raise ValueError("Invalid or duplicate job identity/publication")
            if job.resource_key and any(
                existing.resource_key == job.resource_key for existing, _ in self.pending.values()
            ):
                raise RuntimeError(
                    "A write to this destination is already running; retry when it finishes"
                )
            if self.closed or reserved_bytes < 0 or reserved_bytes > self.max_bytes:
                raise RuntimeError("Job exceeds available execution budget")
            if (
                len(self.pending) >= self.max_tasks
                or self.inflight_bytes + reserved_bytes > self.max_bytes
            ):
                raise RuntimeError("Background queue is full; retry after current work finishes")
            if job.preview_key is not None:
                key = (job.document_id, job.preview_key)
                previous = self.previews.get(key)
                if previous in self.pending:
                    self.pending[previous][0].cancellation.cancel()
                self.previews[key] = job.operation_id
            self.inflight_bytes += reserved_bytes
            self.pending[job.operation_id] = (job, reserved_bytes)

        def run():
            try:
                job.cancellation.check()
                value = function(job.cancellation)
                job.cancellation.check()
                result = Result(job, value)
            except Exception as error:
                result = Result(job, error=error)
            self.results.put(result)

        try:
            self.executor.submit(run, job.priority)
        except Exception:
            with self.lock:
                self.pending.pop(job.operation_id, None)
                self.inflight_bytes -= reserved_bytes
            raise
        return job.operation_id

    def drain(self, documents):
        while not self.results.empty():
            result = self.results.get()
            job = result.job
            with self.lock:
                _, amount = self.pending.pop(job.operation_id)
                self.inflight_bytes -= amount
                preview_current = (
                    job.preview_key is None
                    or self.previews.get((job.document_id, job.preview_key)) == job.operation_id
                )
                if job.preview_key and preview_current:
                    self.previews.pop((job.document_id, job.preview_key), None)
            document = documents.get(job.document_id)
            if document is None or document.closed or job.cancellation.event.is_set():
                self.discarded.append(job.operation_id)
                continue
            if job.publication == "current" and document.generation != job.generation:
                self.discarded.append(job.operation_id)
                continue
            if not preview_current:
                self.discarded.append(job.operation_id)
                continue
            yield result

    def cancel_document(self, document_id):
        with self.lock:
            for job, _ in self.pending.values():
                if job.document_id == document_id:
                    job.cancellation.cancel()

    def shutdown(self):
        with self.lock:
            self.closed = True
            for job, _ in self.pending.values():
                job.cancellation.cancel()
        # Workers observe cancellation at bounded units; no UI calls occur there.
        self.executor.shutdown(wait=False, cancel_futures=False)
