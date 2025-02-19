from app import preprocess_text

def test_preprocess_text():
    assert preprocess_text("Patient diagnosed with HEART FAILURE.") == "patient diagnosed heart failure"
