import torch
import pytest
import yaml

from motion_part_fsl import engine
from motion_part_fsl.smoke import small_config


def test_resume_matches_uninterrupted_run(synthetic_data, tmp_path, monkeypatch):
    metadata, cache_dir = synthetic_data
    config = small_config(metadata, cache_dir, tmp_path / "continuous")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    engine.train_main(["--config", str(path), "--device", "cpu"])
    config["output_dir"] = str(tmp_path / "resumed")
    path.write_text(yaml.safe_dump(config))
    original = engine.evaluate
    calls = 0

    def interrupt_second_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "evaluate", interrupt_second_validation)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        engine.train_main(["--config", str(path), "--device", "cpu"])
    monkeypatch.setattr(engine, "evaluate", original)
    checkpoint = tmp_path / "resumed/last.pt"
    engine.train_main(["--config", str(path), "--device", "cpu", "--resume", str(checkpoint)])
    full = torch.load(tmp_path / "continuous/last.pt", weights_only=True)
    resumed = torch.load(checkpoint, weights_only=True)
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        engine.train_main(["--config", str(path), "--device", "cpu"])


def test_cli_variant_override_resets_guidance_and_enables_optional_hard_loss(synthetic_data, tmp_path):
    config = small_config(*synthetic_data, tmp_path / "a6")
    config["model"]["guidance_mode"] = "full"
    config["training"].update(max_epochs=1, episodes_per_epoch=1)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    engine.train_main(["--config", str(path), "--variant", "A6", "--device", "cpu"])
    state = torch.load(tmp_path / "a6/last.pt", weights_only=True)
    assert state["config"]["loss"]["hard_pair_lambda"] == 0.05
    engine.train_main(["--config", str(path), "--variant", "A0", "--device", "cpu",
                       "--output", str(tmp_path / "a0")])
