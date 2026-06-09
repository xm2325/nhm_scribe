import base64
import json

from PIL import Image

from herbarium_scribe.qwen_vision import (
    image_data_url,
    parse_qwen_vision_output,
    primary_label_vision_messages,
)


def test_image_data_url_resizes_and_encodes_jpeg(tmp_path):
    path = tmp_path / "label.png"
    Image.new("RGB", (400, 200), "white").save(path)

    url, meta = image_data_url(path, max_dimension=100)

    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1])
    assert meta["original_width"] == 400
    assert meta["sent_width"] == 100
    assert meta["sent_height"] == 50


def test_primary_label_messages_keep_transcription_separate_from_fields(tmp_path):
    path = tmp_path / "label.jpg"
    Image.new("RGB", (120, 80), "white").save(path)

    messages, metadata = primary_label_vision_messages([path])

    assert metadata[0]["image_index"] == 1
    content = messages[1]["content"]
    assert content[0]["type"] == "text"
    assert "transcribe every visible character" in content[0]["text"]
    assert "never use a collector number" in content[0]["text"]
    assert "the phrase 'Type Number' alone is not a type status" in content[0]["text"]
    assert any(item["type"] == "image_url" for item in content)


def test_parse_qwen_vision_output_preserves_full_transcription():
    raw = json.dumps({
        "transcriptions": [{
            "image_index": 1,
            "text": "Herb. Kew\nR. E. Holttum\n22 July 1954",
            "uncertain_spans": ["1954"],
        }],
        "fields": {
            "recordedBy": {
                "value": "R. E. Holttum",
                "confidence": 0.91,
                "evidence_span": "R. E. Holttum",
            },
            "eventDate": {
                "value": "1954-07-22",
                "confidence": 0.8,
                "evidence_span": "22 July 1954",
            },
        },
        "observations": ["handwritten"],
    })

    parsed = parse_qwen_vision_output(raw)

    assert parsed["full_transcription"] == "Herb. Kew\nR. E. Holttum\n22 July 1954"
    assert parsed["fields"]["recordedBy"]["value"] == "R. E. Holttum"
    assert parsed["fields"]["eventDate"]["evidence_span"] == "22 July 1954"
    assert parsed["transcriptions"][0]["uncertain_spans"] == ["1954"]
