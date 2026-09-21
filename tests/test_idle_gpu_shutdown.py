from scripts.experiments.shutdown_when_gpu_idle import IdleTimer


def test_requires_ten_continuous_minutes_at_zero():
    timer = IdleTimer(600)
    assert not timer.observe([0], 100)
    assert not timer.observe([0], 699)
    assert timer.observe([0], 700)


def test_gpu_use_or_query_failure_resets_timer():
    timer = IdleTimer(600)
    assert not timer.observe([0, 0], 100)
    assert not timer.observe([0, 1], 500)
    assert not timer.observe([0, 0], 650)
    assert not timer.observe(None, 900)
    assert not timer.observe([0, 0], 1000)
    assert timer.observe([0, 0], 1600)
