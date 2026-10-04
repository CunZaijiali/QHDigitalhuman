import base64
import json
import unittest

import numpy as np

from src.speech.tts import TTSError, VolcengineTTS, synthesize_to_wav


class _FakeResponse:
    def __init__(self, lines, status_code=200, text=""):
        self._lines = lines
        self.status_code = status_code
        self.text = text

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _chunk(samples, code=0):
    pcm = np.asarray(samples, dtype="<i2")
    return json.dumps({"code": code, "data": base64.b64encode(pcm.tobytes()).decode()})


class VolcengineTTSProtocolTest(unittest.TestCase):
    def setUp(self):
        self.backend = VolcengineTTS(api_key="test-key", speaker="test-speaker")

    def test_request_shape(self):
        headers, body = self.backend.build_request("你好")
        self.assertEqual(headers["X-Api-Key"], "test-key")
        self.assertEqual(headers["X-Api-Resource-Id"], "seed-tts-2.0")
        self.assertTrue(headers["X-Api-Request-Id"])
        params = body["req_params"]
        self.assertEqual(params["text"], "你好")
        self.assertEqual(params["speaker"], "test-speaker")
        self.assertEqual(params["audio_params"], {"format": "pcm", "sample_rate": 16000})
        self.assertNotIn("text_type", params)

    def test_ssml_switches_text_type(self):
        _, body = self.backend.build_request("<speak>hi</speak>")
        self.assertEqual(body["req_params"]["text_type"], "ssml")

    def test_missing_api_key_is_rejected(self):
        with self.assertRaises(TTSError) as ctx:
            VolcengineTTS(api_key="", speaker="s")
        self.assertIn("VOLCENGINE_TTS_API_KEY", str(ctx.exception))

    def test_non_https_endpoint_is_rejected(self):
        with self.assertRaises(TTSError):
            VolcengineTTS(api_key="k", speaker="s", endpoint="http://example.com/tts")

    def test_decode_pcm_chunk(self):
        out = self.backend._decode_chunk(_chunk([0, 32767, -32768], code=20000000))
        np.testing.assert_allclose(out, [0.0, 32767 / 32768, -1.0], atol=1e-6)

    def test_decode_reports_error_with_hint(self):
        with self.assertRaises(TTSError) as ctx:
            self.backend._decode_chunk(json.dumps({"code": 55000000, "message": "bad"}))
        self.assertIn("resource/speaker mismatch", str(ctx.exception))

    def test_decode_ignores_noise(self):
        self.assertIsNone(self.backend._decode_chunk("not json at all"))
        self.assertIsNone(self.backend._decode_chunk(json.dumps({"code": 0})))

    def test_synthesize_concatenates_stream_chunks(self):
        import requests

        lines = [_chunk([1000] * 4), "", _chunk([-1000] * 2, code=20000000)]
        original = requests.post
        requests.post = lambda *args, **kwargs: _FakeResponse(lines)
        try:
            out = self.backend.synthesize("hello")
        finally:
            requests.post = original
        self.assertEqual(out.size, 6)
        self.assertTrue(np.allclose(out[:4], 1000 / 32768))

    def test_synthesize_raises_on_http_error(self):
        import requests

        original = requests.post
        requests.post = lambda *args, **kwargs: _FakeResponse([], status_code=401, text="denied")
        try:
            with self.assertRaises(TTSError) as ctx:
                self.backend.synthesize("hello")
        finally:
            requests.post = original
        self.assertIn("401", str(ctx.exception))

    def test_synthesize_rejects_empty_text(self):
        with self.assertRaises(TTSError):
            self.backend.synthesize("   ")


class _StubBackend:
    name = "stub"
    sample_rate = 16000

    def synthesize(self, text):
        return np.full(1600, 0.5, dtype=np.float32)


class SynthesizeToWavTest(unittest.TestCase):
    def test_writes_readable_mono_wav(self):
        import tempfile
        import wave
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.wav"
            synthesize_to_wav("hi", path, backend=_StubBackend())
            with wave.open(str(path), "rb") as stream:
                self.assertEqual(stream.getnchannels(), 1)
                self.assertEqual(stream.getframerate(), 16000)
                self.assertEqual(stream.getnframes(), 1600)


if __name__ == "__main__":
    unittest.main()
