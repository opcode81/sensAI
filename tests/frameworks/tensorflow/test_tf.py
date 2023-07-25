from sensai.util import mark_used


def test_tf():
    import sensai.tensorflow
    mark_used(sensai.tensorflow)
    assert True
