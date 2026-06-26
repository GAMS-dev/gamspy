from __future__ import annotations

import _thread
import signal
import threading
import time

import pytest


@pytest.mark.requires_license
def test_graceful_interrupt(clearlake_built):
    # Simulate a CTRL+C mid-train. interrupt_main() triggers train()'s installed
    # SIGINT handler - the same clean boundary path a real CTRL+C takes when the
    # in-flight solve is not aborted. The Windows aborted-solve path needs a real
    # console event and is validated manually.
    c = clearlake_built
    handler_before = signal.getsignal(signal.SIGINT)

    def fire():
        time.sleep(1.5)
        _thread.interrupt_main()

    threading.Thread(target=fire, daemon=True).start()
    result = c.sddp.train(n_iter=50)

    assert result.stop_reason == "interrupted"
    assert 0 < result.iterations_run < 50

    # The partially-trained policy is usable.
    pol = c.sddp.policy(stage="mar", state=180, noise=100, report=[c.rel, c.lev])
    assert isinstance(pol.approx_cost_to_go, float)

    # The original SIGINT handler is restored after train().
    assert signal.getsignal(signal.SIGINT) is handler_before
