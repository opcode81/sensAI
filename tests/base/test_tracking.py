from sensai.util import mark_used


def test_tracking():
    import sensai.tracking
    mark_used(sensai.tracking)
    assert True
