from workloads import WORKLOADS


def test_eight_workloads():
    assert len(WORKLOADS) == 8


def test_required_keys():
    for w in WORKLOADS:
        assert set(w.keys()) >= {"label", "M", "K", "N", "E", "topk"}


def test_shape_values_match_spec():
    expected = [
        ("decode",  1,    2048, 1024, 8,  2),
        ("decode",  64,   2048, 1024, 8,  2),
        ("decode",  1,    2048, 1024, 64, 6),
        ("decode",  64,   2048, 1024, 64, 6),
        ("prefill", 512,  2048, 1024, 8,  2),
        ("prefill", 4096, 2048, 1024, 8,  2),
        ("prefill", 512,  2048, 1024, 64, 6),
        ("prefill", 4096, 2048, 1024, 64, 6),
    ]
    actual = [(w["label"], w["M"], w["K"], w["N"], w["E"], w["topk"]) for w in WORKLOADS]
    assert actual == expected
