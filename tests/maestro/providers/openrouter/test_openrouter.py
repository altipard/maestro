"""Tests for OpenRouter provider registration and configuration."""

from maestro.providers.registry import _registry


class TestOpenRouterRegistry:
    def test_completer_registered(self):
        assert "openrouter" in _registry
        assert "completer" in _registry["openrouter"]

    def test_renderer_registered(self):
        assert "renderer" in _registry["openrouter"]

    def test_synthesizer_registered(self):
        assert "synthesizer" in _registry["openrouter"]

    def test_transcriber_registered(self):
        assert "transcriber" in _registry["openrouter"]

    def test_completer_is_subclass_of_openai(self):
        from maestro.providers.openai.completer import Completer as OpenAICompleter
        from maestro.providers.openrouter.completer import Completer as ORCompleter

        assert issubclass(ORCompleter, OpenAICompleter)

    def test_completer_default_url(self):
        from maestro.providers.openrouter.completer import Completer

        c = Completer(token="test", model="test/model")
        assert "openrouter.ai" in c._url

    def test_completer_custom_url(self):
        from maestro.providers.openrouter.completer import Completer

        c = Completer(url="https://custom.proxy/v1", token="test", model="test/model")
        assert c._url == "https://custom.proxy/v1"

    def test_renderer_default_url(self):
        from maestro.providers.openrouter.renderer import Renderer

        r = Renderer(token="test", model="test/model")
        assert "openrouter.ai" in r._url

    def test_synthesizer_default_url(self):
        from maestro.providers.openrouter.synthesizer import Synthesizer

        s = Synthesizer(token="test", model="test/model")
        assert "openrouter.ai" in s._url

    def test_transcriber_default_url(self):
        from maestro.providers.openrouter.transcriber import Transcriber

        t = Transcriber(token="test", model="test/model")
        assert "openrouter.ai" in t._url


class TestAudioFormatDetection:
    def test_wav_from_content_type(self):
        from maestro.core.models import FileData
        from maestro.providers.openrouter.transcriber import _detect_audio_format

        f = FileData(name="", content=b"", content_type="audio/x-wav")
        assert _detect_audio_format(f) == "wav"

    def test_mp3_from_content_type(self):
        from maestro.core.models import FileData
        from maestro.providers.openrouter.transcriber import _detect_audio_format

        f = FileData(name="", content=b"", content_type="audio/mpeg")
        assert _detect_audio_format(f) == "mp3"

    def test_format_from_filename(self):
        from maestro.core.models import FileData
        from maestro.providers.openrouter.transcriber import _detect_audio_format

        f = FileData(name="recording.ogg", content=b"", content_type="")
        assert _detect_audio_format(f) == "ogg"

    def test_aif_to_aiff(self):
        from maestro.core.models import FileData
        from maestro.providers.openrouter.transcriber import _detect_audio_format

        f = FileData(name="sound.aif", content=b"", content_type="")
        assert _detect_audio_format(f) == "aiff"

    def test_default_wav(self):
        from maestro.core.models import FileData
        from maestro.providers.openrouter.transcriber import _detect_audio_format

        f = FileData(name="", content=b"", content_type="")
        assert _detect_audio_format(f) == "wav"
