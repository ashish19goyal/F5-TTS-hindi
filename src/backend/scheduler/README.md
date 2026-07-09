# Scheduler

This package provides a scheduler abstraction for dispatching chunks to inference workers.

- `LocalThreadScheduler` — uses `concurrent.futures.ThreadPoolExecutor` for local parallelism.
- `RayScheduler` — uses Ray for distributed execution (optional dependency).

Usage example:

```python
from scheduler import LocalThreadScheduler

scheduler = LocalThreadScheduler(max_workers=4)
results = scheduler.schedule(chunks, inference.infer)
```
