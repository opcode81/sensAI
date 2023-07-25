from sensai.util import mark_used


def test_util():
    import sensai.util
    mark_used(sensai.util)
    assert True
