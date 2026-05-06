from mrbrains.config import apply_overrides, deep_update, parse_override


def test_parse_override_scalar():
    assert parse_override("training.epochs=5") == {"training": {"epochs": 5}}


def test_parse_override_list():
    assert parse_override("inference.tta_flips=[[3]]") == {"inference": {"tta_flips": [[3]]}}


def test_parse_override_bool():
    assert parse_override("training.amp=false") == {"training": {"amp": False}}


def test_deep_update_recursively_merges():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_update(base, {"a": {"c": 99}, "d": 4})
    assert out == {"a": {"b": 1, "c": 99}, "d": 4}


def test_apply_overrides_chain():
    base = {"training": {"epochs": 1, "lr": 0.001}}
    out = apply_overrides(base, ["training.epochs=10", "training.lr=0.0005"])
    assert out["training"] == {"epochs": 10, "lr": 0.0005}


def test_apply_overrides_no_tokens_is_identity():
    base = {"x": 1}
    assert apply_overrides(base, None) == base
    assert apply_overrides(base, []) == base
