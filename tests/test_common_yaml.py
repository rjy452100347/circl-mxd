import yaml

from src.utils import common


def test_save_yaml_keeps_previous_file_visible_until_atomic_replace(
    tmp_path, monkeypatch
):
    target = tmp_path / "config.yaml"
    target.write_text("generation: old\n", encoding="utf-8")
    original_dump = common.yaml.dump
    observed = []

    def inspecting_dump(data, stream, **kwargs):
        observed.append(yaml.safe_load(target.read_text(encoding="utf-8")))
        return original_dump(data, stream, **kwargs)

    monkeypatch.setattr(common.yaml, "dump", inspecting_dump)

    common.save_yaml({"generation": "new", "items": [1, 2]}, target)

    assert observed == [{"generation": "old"}]
    assert yaml.safe_load(target.read_text(encoding="utf-8")) == {
        "generation": "new",
        "items": [1, 2],
    }
    assert list(tmp_path.glob(".*.tmp")) == []


def test_save_yaml_serialization_failure_preserves_previous_file(
    tmp_path, monkeypatch
):
    target = tmp_path / "config.yaml"
    target.write_text("generation: old\n", encoding="utf-8")

    def failing_dump(_data, stream, **_kwargs):
        stream.write("generation:")
        raise RuntimeError("simulated interrupted serialization")

    monkeypatch.setattr(common.yaml, "dump", failing_dump)

    try:
        common.save_yaml({"generation": "new"}, target)
    except RuntimeError as exc:
        assert str(exc) == "simulated interrupted serialization"
    else:
        raise AssertionError("save_yaml should propagate serialization errors")

    assert yaml.safe_load(target.read_text(encoding="utf-8")) == {
        "generation": "old"
    }
    assert list(tmp_path.glob(".*.tmp")) == []
