"""Image-label requests: token counts, windowing, side wording, request layout, answer checks. No network."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gpt_image_labels", ROOT / "scripts/gpt_image_labels.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_image_tokens_follow_the_documented_examples():
    assert MODULE.image_tokens(1024, 1024) == 1229
    assert MODULE.image_tokens(2048, 2048) == 3000
    assert MODULE.image_tokens(512, 512) == 308
    assert MODULE.image_tokens(336, 336) == 146


def test_series_windowing_is_shared_and_blank_pixels_become_black():
    first = np.linspace(0, 1000, 100, dtype=np.float32).reshape(10, 10)
    second = first.copy()
    second[0, 0] = np.nan
    scaled, window = MODULE.to_uint8([first, second])
    assert all(image.dtype == np.uint8 for image in scaled)
    assert scaled[1][0, 0] == 0 and scaled[0][-1, -1] == 255
    np.testing.assert_array_equal(scaled[0][1:], scaled[1][1:])
    assert window[0] < window[1]


def test_side_wording_places_medial_correctly():
    left = MODULE.study_header("left", "dicom_laterality", 5, 100)
    right = MODULE.study_header("right", "geometry", 5, 100)
    assert "medial side is on the image left" in left and "first slices are medial" in left
    assert "medial side is on the image right" in right and "first slices are lateral" in right
    assert "fibula" in MODULE.study_header(None, None, 5, 100)


def make_pack(tmp_path):
    pack_dir = tmp_path / "study"
    pack_dir.mkdir()
    series = []
    for number, plane in ((1, "Sagittal"), (2, "Coronal")):
        images = []
        for index in (1, 2):
            label = f"S{number}-{index:02d}"
            Image.fromarray(np.full((8, 8), 40 * index, dtype=np.uint8)).save(pack_dir / f"{label}.png")
            images.append({"label": label, "file": f"{label}.png", "shape": [8, 8]})
        series.append({"number": number, "series": f"series{number}", "plane": plane, "description": "PD FS",
                       "fluid_sensitive": True, "fat_suppressed": True, "slice_spacing_mm": 3.0,
                       "ordering": "physical_position", "orientation_standardized": True, "window": [0, 1],
                       "warnings": [], "images": images})
    (pack_dir / "pack.json").write_text(json.dumps({"study": "study", "knee_side": "left", "knee_side_source": "geometry",
                                                    "series": series, "image_count": 4}))
    return pack_dir


def test_request_interleaves_labels_and_full_images_with_a_strict_format(tmp_path):
    body = MODULE.build_request(make_pack(tmp_path), "high")
    content = body["input"][1]["content"]
    kinds = [part["type"] for part in content]
    assert kinds == ["input_text"] + (["input_text"] + ["input_text", "input_image"] * 2) * 2
    assert [part["text"] for part in content if part["type"] == "input_text"][2] == "S1-01"
    assert all(part["detail"] == "high" and part["image_url"].startswith("data:image/png;base64,")
               for part in content if part["type"] == "input_image")
    assert body["input"][0]["content"][0]["text"] == MODULE.PROMPT.read_text()
    assert body["store"] is False and body["reasoning"] == {"effort": "high"}
    schema = body["text"]["format"]
    assert schema["strict"] is True and schema["schema"]["properties"]["findings"]["required"] == list(MODULE.FINDINGS)
    assert "report" not in json.dumps(content).lower()


def answer(**changes):
    findings = {key: {"probability": 10, "assessable": True, "evidence": ["S1-01"], "note": "n"} for key in MODULE.FINDINGS}
    findings["effusion"]["probability"] = 75
    for key, value in changes.items():
        findings[key] = value
    return json.dumps({"findings": findings, "image_quality_note": "ok"})


def test_valid_answers_become_probabilities_and_bad_ones_are_refused():
    row = MODULE.parse_reply("study", answer())
    assert row["Effusion"] == 0.75 and row["ACL"] == 0.10 and row["Baker's__assessable"] is True
    with pytest.raises(ValueError, match="integer from 0 to 100"):
        MODULE.parse_reply("study", answer(acl={"probability": 150, "assessable": True, "evidence": [], "note": ""}))
    broken = json.loads(answer())
    del broken["findings"]["fracture"]
    with pytest.raises(ValueError, match="twelve"):
        MODULE.parse_reply("study", json.dumps(broken))


def test_reply_text_and_cost_come_from_the_saved_response():
    response = {"status": "completed", "output": [{"type": "reasoning"}, {"type": "message", "content": [
        {"type": "output_text", "text": answer()}]}],
        "usage": {"input_tokens": 1_000_000, "input_tokens_details": {"cached_tokens": 100_000},
                  "output_tokens": 20_000, "output_tokens_details": {"reasoning_tokens": 18_000}}}
    assert json.loads(MODULE.reply_text(response))["image_quality_note"] == "ok"
    cost = MODULE.usage_cost(response["usage"])
    assert cost["reasoning_tokens"] == 18_000 and cost["standard_usd"] == pytest.approx(9.0 + 0.1 + 1.0)


def test_run_refuses_without_explicit_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["gpt_image_labels.py", "run", "--studies", "x", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="without --yes"):
        MODULE.main()


def test_named_reply_folders_get_their_own_labels_file(tmp_path):
    assert MODULE.labels_path(tmp_path, "replies") == tmp_path / "labels.csv"
    assert MODULE.labels_path(tmp_path, "replies-max") == tmp_path / "labels-replies-max.csv"


def test_studies_with_saved_replies_are_not_sent_again(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    assert MODULE.pending_studies(tmp_path, ["a", "b", "c"]) == ["b", "c"]


def test_large_images_shrink_to_the_size_cap_and_small_ones_stay():
    assert MODULE.shrink(np.zeros((1024, 1024), dtype=np.uint8), 512).shape == (512, 512)
    assert MODULE.shrink(np.zeros((640, 1280), dtype=np.uint8), 512).shape == (256, 512)
    small = np.full((400, 400), 7, dtype=np.uint8)
    assert MODULE.shrink(small, 512) is small and MODULE.shrink(small, None) is small


def test_claude_image_tokens_follow_the_documented_patch_rule():
    assert MODULE.claude_image_tokens(512, 512) == 19 * 19
    assert MODULE.claude_image_tokens(28, 28) == 1
    assert MODULE.claude_image_tokens(3000, 3000) <= MODULE.CLAUDE_MAX_IMAGE_TOKENS


def test_claude_request_carries_effort_format_and_labelled_images(tmp_path):
    pack_dir = make_pack(tmp_path)
    body = MODULE.build_claude_request(pack_dir, "high")
    assert body["model"] == "claude-opus-5-5" and body["system"] == MODULE.PROMPT.read_text()
    assert body["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": MODULE.CLAUDE_SCHEMA}}
    content = body["messages"][0]["content"]
    assert [part["type"] for part in content] == ["text"] + (["text"] + ["text", "image"] * 2) * 2 + ["text"]
    assert content[-1]["text"] == MODULE.CLAUDE_FORMAT_NOTE
    assert content[2]["text"] == "S1-01" and content[3]["source"]["media_type"] == "image/png"
    files = {f"S{s}-0{i}": f"file_{s}{i}" for s in (1, 2) for i in (1, 2)}
    linked = MODULE.build_claude_request(pack_dir, "high", files)["messages"][0]["content"]
    assert linked[3]["source"] == {"type": "file", "file_id": "file_11"}
    assert "report" not in json.dumps(content).lower()


def test_upload_body_has_the_expiry_field_and_the_image_bytes():
    body, content_type = MODULE.multipart_body({"expires_in_seconds": "172800"}, "x.png", b"PNGDATA", "image/png")
    boundary = content_type.split("boundary=")[1]
    assert b'name="expires_in_seconds"\r\n\r\n172800' in body and b'filename="x.png"' in body
    assert b"Content-Type: image/png\r\n\r\nPNGDATA\r\n" in body and body.endswith(f"--{boundary}--\r\n".encode())


def test_claude_replies_and_costs_are_read_correctly():
    response = {"type": "message", "stop_reason": "end_turn",
                "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": answer()}],
                "usage": {"input_tokens": 1_000_000, "output_tokens": 100_000}}
    assert json.loads(MODULE.reply_text(response))["image_quality_note"] == "ok"
    assert MODULE.claude_usage_cost(response["usage"])["standard_usd"] == pytest.approx(6.0)
    with pytest.raises(ValueError, match="stop reason"):
        MODULE.reply_text({"type": "message", "content": [], "stop_reason": "max_tokens"})


def test_claude_list_answers_parse_like_the_nested_form_and_refuse_duplicates():
    nested = json.loads(answer())
    listed = {"findings": [{"finding": key, **value} for key, value in nested["findings"].items()],
              "image_quality_note": "ok"}
    assert MODULE.parse_reply("study", json.dumps(listed)) == MODULE.parse_reply("study", answer())
    listed["findings"].append(dict(listed["findings"][0]))
    with pytest.raises(ValueError, match="more than once"):
        MODULE.parse_reply("study", json.dumps(listed))


def test_a_failed_upload_deletes_the_images_already_uploaded(monkeypatch, tmp_path):
    def fake_request(url, headers, data=None, method="POST", attempts=6, timeout=3600):
        assert timeout == MODULE.FILES_TIMEOUT
        if b'S2-02.png"' in data:
            raise RuntimeError("API error 429 from files")
        return {"id": "file_" + data.split(b'filename="')[1].split(b'"')[0].decode()}

    deleted = []
    monkeypatch.setattr(MODULE, "request_json", fake_request)
    monkeypatch.setattr(MODULE, "delete_files", lambda file_ids, key: deleted.extend(file_ids))
    with pytest.raises(RuntimeError, match="429"):
        MODULE.upload_pack(make_pack(tmp_path), "key")
    assert sorted(deleted) == ["file_study-S1-01.png", "file_study-S1-02.png", "file_study-S2-01.png"]


def test_dropped_connections_are_retried_and_reported_when_they_persist(monkeypatch):
    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"ok": true}'

    calls, waits = [], []

    def flaky(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise ConnectionResetError("reset by peer")
        return Reply()

    monkeypatch.setattr(MODULE.time, "sleep", waits.append)
    monkeypatch.setattr(MODULE.urllib.request, "urlopen", flaky)
    assert MODULE.request_json("https://example.invalid/x", {}) == {"ok": True}
    assert len(calls) == 2 and waits == [20]
    monkeypatch.setattr(MODULE.urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(RuntimeError, match="Connection problem"):
        MODULE.request_json("https://example.invalid/x", {}, attempts=2)


def test_one_failed_study_does_not_stop_the_others(monkeypatch, tmp_path, capsys):
    for study in ("a", "b"):
        make_pack(tmp_path).rename(tmp_path / f"packs-{study}")
    (tmp_path / "packs").mkdir()
    for study in ("a", "b"):
        (tmp_path / f"packs-{study}").rename(tmp_path / "packs" / study)

    def fake_request(url, headers, data=None, method="POST", attempts=6, timeout=3600):
        if json.loads(data)["study"] == "a":
            raise RuntimeError("API error 400 from messages")
        return {"stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}, "content": []}

    monkeypatch.setattr(MODULE, "api_key", lambda name: "key")
    monkeypatch.setattr(MODULE, "build_claude_request", lambda pack_dir, effort, file_ids=None, model=None: {"study": pack_dir.name})
    monkeypatch.setattr(MODULE, "request_json", fake_request)
    monkeypatch.setattr(sys, "argv", ["gpt_image_labels.py", "run", "--yes", "--provider", "anthropic", "--workers", "2",
                                      "--studies", "a", "b", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="1 of 2 studies failed"):
        MODULE.main()
    assert (tmp_path / "replies" / "b.json").exists() and not (tmp_path / "replies" / "a.json").exists()
    assert '"study": "a", "error": "API error 400' in capsys.readouterr().out


def test_answers_are_saved_before_the_uploaded_images_are_deleted(monkeypatch, tmp_path, capsys):
    (tmp_path / "packs").mkdir()
    make_pack(tmp_path).rename(tmp_path / "packs" / "a")

    def fake_request(url, headers, data=None, method="POST", attempts=6, timeout=3600):
        if url == MODULE.CLAUDE_FILES_URL:
            return {"id": "file_" + data.split(b'filename="')[1].split(b'"')[0].decode()}
        if method == "DELETE":
            if url.endswith("S1-01.png"):
                raise RuntimeError("Connection problem with files")
            return {}
        return {"stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}, "content": []}

    monkeypatch.setattr(MODULE, "CLAUDE_REQUEST_LIMIT", 0)
    monkeypatch.setattr(MODULE, "api_key", lambda name: "key")
    monkeypatch.setattr(MODULE, "request_json", fake_request)
    monkeypatch.setattr(sys, "argv", ["gpt_image_labels.py", "run", "--yes", "--provider", "anthropic",
                                      "--studies", "a", "--out", str(tmp_path)])
    MODULE.main()
    assert (tmp_path / "replies" / "a.json").exists()
    line = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert line["uploaded_images"] == 4 and line["files_left"] == 1


def test_deletes_report_what_is_left_and_count_missing_files_as_gone(monkeypatch):
    def fake_request(url, headers, data=None, method="POST", attempts=6, timeout=3600):
        assert method == "DELETE" and timeout == MODULE.FILES_TIMEOUT
        if url.endswith("/gone"):
            raise RuntimeError("API error 404 from files: not found")
        if url.endswith("/stuck"):
            raise RuntimeError("Connection problem with files")
        return {}

    monkeypatch.setattr(MODULE, "request_json", fake_request)
    assert MODULE.delete_files(["ok", "gone", "stuck"], "key") == ["stuck"]


def test_cleanup_deletes_only_these_studies_leftover_images(monkeypatch, capsys):
    pages = {None: {"data": [{"id": "f1", "filename": "abcdefabcdef-S1-01.png"}, {"id": "f2", "filename": "other-x.png"}],
                    "has_more": True, "last_id": "f2"},
             "f2": {"data": [{"id": "f3", "filename": "abcdefabcdef-S2-01.png"}], "has_more": False, "last_id": "f3"}}
    deleted = []

    def fake_request(url, headers, data=None, method="POST", attempts=6, timeout=3600):
        if method == "GET":
            return pages[url.split("after_id=")[1] if "after_id=" in url else None]
        deleted.append(url.rsplit("/", 1)[1])
        return {}

    monkeypatch.setattr(MODULE, "api_key", lambda name: "key")
    monkeypatch.setattr(MODULE, "request_json", fake_request)
    monkeypatch.setattr(sys, "argv", ["gpt_image_labels.py", "cleanup", "--studies", "1.2.3.abcdefabcdef"])
    MODULE.main()
    assert sorted(deleted) == ["f1", "f3"]
    assert json.loads(capsys.readouterr().out) == {"uploaded_files_found": 2, "deleted": 2, "left": 0}


def test_another_claude_model_is_named_in_the_request_and_priced_at_its_own_rate(tmp_path):
    body = MODULE.build_claude_request(make_pack(tmp_path), "high", model="claude-fable-5-1")
    assert body["model"] == "claude-fable-5-1"
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000}
    assert MODULE.claude_usage_cost(usage, "claude-fable-5-1")["standard_usd"] == pytest.approx(15.0)
    assert MODULE.claude_usage_cost({**usage, "cache_read_input_tokens": 1_000_000}, "claude-fable-5-1")["standard_usd"] \
        == pytest.approx(15.25)


def test_flex_requests_ask_for_the_tier_and_are_charged_half(tmp_path):
    pack_dir = make_pack(tmp_path)
    assert "service_tier" not in MODULE.build_request(pack_dir, "high")
    assert MODULE.build_request(pack_dir, "high", service_tier="flex")["service_tier"] == "flex"
    usage = {"input_tokens": 1_000_000, "output_tokens": 20_000}
    assert MODULE.usage_cost(usage, "flex")["charged_usd"] == pytest.approx(5.5)
    assert MODULE.usage_cost(usage, "default")["charged_usd"] == MODULE.usage_cost(usage)["standard_usd"] == pytest.approx(11.0)


def test_gemini_request_interleaves_labels_and_images_with_a_json_schema(tmp_path):
    pack_dir = make_pack(tmp_path)
    body = MODULE.build_gemini_request(pack_dir, "medium")
    parts = body["contents"][0]["parts"]
    kinds = ["inlineData" if "inlineData" in part else "text" for part in parts]
    assert kinds == ["text"] + (["text"] + ["text", "inlineData"] * 2) * 2
    assert parts[2]["text"] == "S1-01" and parts[3]["inlineData"]["mimeType"] == "image/png"
    assert body["systemInstruction"]["parts"][0]["text"] == MODULE.PROMPT.read_text()
    config = body["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingLevel": "medium"} and config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"]["properties"]["findings"]["required"] == list(MODULE.FINDINGS)
    uris = {f"S{s}-0{i}.png": f"https://example/files/{s}{i}" for s in (1, 2) for i in (1, 2)}
    linked = MODULE.build_gemini_request(pack_dir, "medium", uris)["contents"][0]["parts"]
    assert linked[3] == {"fileData": {"mimeType": "image/png", "fileUri": "https://example/files/11"}}
    assert "report" not in json.dumps(parts).lower()
    budget = MODULE.build_gemini_request(pack_dir, "high", thinking_budget=24000)["generationConfig"]["thinkingConfig"]
    assert budget == {"thinkingBudget": 24000}  # the API accepts a level or a budget, never both


def test_gemini_replies_skip_thoughts_and_count_thinking_as_output():
    response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": "working it out", "thought": True}, {"text": answer()}]}}],
        "usageMetadata": {"promptTokenCount": 1_000_000, "candidatesTokenCount": 100_000, "thoughtsTokenCount": 100_000}}
    assert MODULE.parse_reply("study", MODULE.reply_text(response))["Effusion"] == 0.75
    cost = MODULE.gemini_usage_cost(response["usageMetadata"])
    assert cost == {"input_tokens": 1_000_000, "output_tokens": 200_000, "reasoning_tokens": 100_000, "standard_usd": 4.0}
    with pytest.raises(ValueError, match="SAFETY"):
        MODULE.reply_text({"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]})


def test_tissue_crop_keeps_one_box_per_series_and_drops_only_empty_end_slices():
    rng = np.random.default_rng(0)

    def noise():  # bright speckle (mean about 20 of 255) that blurs to a flat level
        return rng.integers(0, 40, (100, 100)).astype(np.uint8)

    knee = noise()
    knee[30:70, 40:70] = 200
    kept, box = MODULE.tissue_crop([noise(), knee, noise(), knee.copy(), noise()])
    assert kept == [1, 2, 3]  # empty slices go only at the ends
    top, left, bottom, right = box
    assert top <= 30 and left <= 40 and bottom >= 70 and right >= 70  # the tissue is inside, with a margin
    assert top >= 20 and left >= 30 and bottom <= 80 and right <= 80  # noise does not stretch the box
    full = np.full((100, 100), 90, np.uint8)
    assert MODULE.tissue_crop([full, full]) == ([0, 1], None)  # tissue fills the frame: nothing to trim
    assert MODULE.tissue_crop([knee, np.zeros((50, 50), np.uint8)]) == ([0, 1], None)  # mixed shapes: left as is
    assert MODULE.tissue_crop([noise(), noise()]) == ([0, 1], None)  # no tissue anywhere: left as is
