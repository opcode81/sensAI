from sensai.util import markUsed


def test_geoanalytics_geopandas():
    import sensai.geoanalytics.geopandas
    markUsed(sensai.geoanalytics.geopandas)
    assert True