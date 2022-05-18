from sensai.util import markUsed


def test_tf():
    import sensai.tensorflow
    markUsed(sensai.tensorflow)
    assert True
