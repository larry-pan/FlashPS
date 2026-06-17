import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flashps_router import (  # noqa: E402
    RequestSpec, classify, resolve_seqlen, render_config, build_request,
    DEFAULT_FULL_SEQLEN, MIN_EDIT_SEQLEN, SERVICE_ID,
)

FULL = DEFAULT_FULL_SEQLEN
BASE_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "configs", "flux_inpaint_base.yml")


def _req(**kw):
    kw.setdefault("prompt", "p")
    kw.setdefault("image_path", "/tmp/x.png")
    return RequestSpec(**kw)


def test_explicit_edit_flag_overrides_auto():
    assert classify(_req(edit=True), FULL) == "edit"
    assert classify(_req(edit=False, mask_image_path="/tmp/m.png", mask_seq_length=100), FULL) == "non_edit"


def test_auto_no_mask_is_non_edit():
    assert classify(_req(), FULL) == "non_edit"


def test_auto_small_mask_is_edit_large_is_non_edit():
    assert classify(_req(mask_image_path="/tmp/m.png", mask_seq_length=int(0.2 * FULL)), FULL) == "edit"
    assert classify(_req(mask_image_path="/tmp/m.png", mask_seq_length=int(0.9 * FULL)), FULL) == "non_edit"


def test_resolve_seqlen_non_edit_is_full():
    assert resolve_seqlen(_req(edit=False), "non_edit", FULL) == FULL


def test_resolve_seqlen_edit_uses_mask_value_clamped():
    assert resolve_seqlen(_req(mask_seq_length=500), "edit", FULL) == 500
    assert resolve_seqlen(_req(mask_seq_length=1), "edit", FULL) == MIN_EDIT_SEQLEN
    assert resolve_seqlen(_req(mask_seq_length=99999), "edit", FULL) == FULL


def test_resolve_seqlen_edit_without_mask_defaults_to_quarter():
    assert resolve_seqlen(_req(edit=True), "edit", FULL) == FULL // 4


def test_render_config(tmp_path):
    edit_path = render_config(BASE_CONFIG, "edit", 800, str(tmp_path), 0)
    cfg = yaml.safe_load(open(edit_path))
    assert cfg["use_cached_kv"] is True and cfg["generated_seqlen"] == 800 and cfg["test_seqlen"] is True
    non_path = render_config(BASE_CONFIG, "non_edit", FULL, str(tmp_path), 1)
    assert yaml.safe_load(open(non_path))["use_cached_kv"] is False


def test_build_request_edit(tmp_path):
    service_id, inputs, route = build_request(
        _req(edit=True, mask_seq_length=600, mask_image_path="/tmp/m.png"), BASE_CONFIG, str(tmp_path), 0)
    assert service_id == SERVICE_ID and route == "edit" and inputs["mask_seq_length"] == 600
    assert yaml.safe_load(open(inputs["edit_config_path"]))["use_cached_kv"] is True


def test_build_request_non_edit(tmp_path):
    _, inputs, route = build_request(_req(edit=False), BASE_CONFIG, str(tmp_path), 1)
    assert route == "non_edit" and inputs["mask_seq_length"] == FULL
    assert yaml.safe_load(open(inputs["edit_config_path"]))["use_cached_kv"] is False
