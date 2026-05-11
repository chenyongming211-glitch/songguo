from __future__ import annotations

from pathlib import Path
import re
from uuid import uuid4

from pydantic import BaseModel


class VoiceDraft(BaseModel):
    transcript: str = ""
    confidence: float = 0.0
    needs_confirmation: bool = True


class DeterministicASRProvider:
    """Local ASR fallback for text fixtures and offline development.

    It accepts UTF-8 fixtures shaped like:
    TRANSCRIPT: ...
    """

    def recognize(self, content: bytes, *, filename: str, content_type: str = "") -> VoiceDraft:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return VoiceDraft()
        transcript = _extract_transcript(text)
        return VoiceDraft(
            transcript=transcript,
            confidence=0.92 if transcript else 0.0,
            needs_confirmation=not transcript,
        )


class OpenAICompatibleASRProvider:
    """Real ASR provider for OpenAI-compatible /audio/transcriptions endpoints."""

    def __init__(self, *, model: str = "whisper-1") -> None:
        self.model = model

    def recognize(self, content: bytes, *, filename: str, content_type: str = "") -> VoiceDraft:
        import requests

        from songguo.backend.services.learning.real_model_client import load_real_model_config

        config = load_real_model_config()
        if not config.api_key:
            raise RuntimeError("LLM_API_KEY is not configured")
        endpoint = f"{config.base_url.rstrip('/')}/audio/transcriptions"
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {config.api_key}"},
            data={"model": self.model},
            files={
                "file": (
                    filename or "voice.mp3",
                    content,
                    content_type or "audio/mpeg",
                )
            },
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        transcript = str(payload.get("text") or "").strip()
        return VoiceDraft(
            transcript=transcript,
            confidence=0.9 if transcript else 0.0,
            needs_confirmation=not transcript,
        )


class VoiceInputService:
    def __init__(self, *, artifact_root: Path, asr_provider: object | None = None) -> None:
        self.artifact_root = artifact_root
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.asr_provider = asr_provider or DeterministicASRProvider()

    async def recognize_submission_draft_async(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> tuple[str, VoiceDraft]:
        audio_path = self._save_artifact(filename, content)
        draft = self._recognize(content, filename=filename, content_type=content_type)
        return audio_path, draft

    def _recognize(self, content: bytes, *, filename: str, content_type: str) -> VoiceDraft:
        try:
            return self.asr_provider.recognize(
                content,
                filename=filename,
                content_type=content_type,
            )
        except Exception:
            return VoiceDraft()

    def _save_artifact(self, filename: str, content: bytes) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename or "voice.mp3")
        target = self.artifact_root / f"{uuid4().hex}_{safe_name}"
        target.write_bytes(content)
        return str(target)


def _extract_transcript(text: str) -> str:
    match = re.search(r"^TRANSCRIPT\s*:\s*(.*)$", text.strip(), re.IGNORECASE | re.DOTALL)
    return (match.group(1) if match else text).strip()
