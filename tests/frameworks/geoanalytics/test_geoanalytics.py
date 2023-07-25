from sensai.util import mark_used


def test_geoanalytics_geopandas():
    import sensai.geoanalytics.geopandas
    mark_used(sensai.geoanalytics.geopandas)
    assert True