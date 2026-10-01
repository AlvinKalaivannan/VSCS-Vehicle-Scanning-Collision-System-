"""P6-T2 required tests: the queue is bounded, drops the stalest, and latency stays bounded."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from vscs.stream.queues import DropOldestQueue


def test_bounded_and_drops_the_stalest():
    q = DropOldestQueue(3)
    assert [q.put(i) for i in range(3)] == [None, None, None]
    assert q.put(3) == 0 and q.put(4) == 1  # the oldest go first
    assert len(q) == 3 and q.dropped == 2 and q.puts == 5 and q.max_depth == 3
    assert [q.get(0), q.get(0), q.get(0)] == [2, 3, 4]  # the newest survive, in order
    assert q.get(0) is None


def test_close_drains_then_returns_none_and_refuses_puts():
    q = DropOldestQueue(2)
    q.put("a")
    q.close()
    assert q.get(0) == "a" and q.get(0) is None
    with pytest.raises(RuntimeError):
        q.put("b")


def test_bad_size_is_rejected():
    with pytest.raises(ValueError):
        DropOldestQueue(0)


def _simulate(producer_period_ms, service_ms, maxsize, n_frames):
    """Discrete-event simulation: one producer, one consumer, exact times, no threads.

    Returns the age (ms) of each frame when the consumer *finishes* it.
    """
    q = DropOldestQueue(maxsize)
    arrivals = [i * producer_period_ms for i in range(n_frames)]
    ages, t_free, k = [], 0.0, 0
    while k < len(arrivals) or len(q):
        # Everything that has arrived by the time the consumer is free goes in first.
        while k < len(arrivals) and arrivals[k] <= t_free:
            q.put(arrivals[k])
            k += 1
        item = q.get(0)
        if item is None:  # idle until the next arrival
            t_free = arrivals[k]
            continue
        t_free = max(t_free, item) + service_ms
        ages.append(t_free - item)
    return np.array(ages), q.dropped


def test_latency_does_not_grow_under_2x_overload():
    """Frames every 10 ms into a 20 ms stage: twice what it can handle."""
    ages, dropped = _simulate(producer_period_ms=10, service_ms=20, maxsize=2, n_frames=600)
    assert dropped > 250  # about half the frames must go: the cost of overload is visible
    # Bounded: no frame waits longer than the queue's worth of service plus its own.
    assert ages.max() <= (2 + 1) * 20
    # Not growing: late frames are no older than early ones (after warm-up). The last few
    # are excluded: when the input stops they drain from a full queue, an end effect.
    assert ages[-120:-20].mean() <= ages[20:120].mean() + 1e-9
    assert ages[20:-20].max() - ages[20:-20].min() <= 1e-9  # steady: exactly one age


def test_an_unbounded_queue_would_grow_for_contrast():
    """Why the drop matters: with room for every frame, age climbs without limit."""
    ages, dropped = _simulate(producer_period_ms=10, service_ms=20, maxsize=10_000, n_frames=600)
    assert dropped == 0
    assert ages[-1] > 10 * ages[10]


def test_threads_can_share_it():
    q = DropOldestQueue(4)
    got = []

    def consume():
        while (item := q.get(timeout=1.0)) is not None:
            got.append(item)

    t = threading.Thread(target=consume)
    t.start()
    for i in range(1000):
        q.put(i)
    q.close()
    t.join(timeout=5)
    assert not t.is_alive()
    assert got == sorted(got) and got[-1] == 999  # in order; the newest is never dropped
    assert len(got) + q.dropped == 1000
