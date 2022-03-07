from app.main import file_clean_up


def test_file_clean_up():
    assert file_clean_up([]) is None
