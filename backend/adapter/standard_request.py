from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.runtime.attachment_types import NormalizedAttachment
from backend.services.client_profiles import (
    CLAUDE_CODE_OPENAI_PROFILE,
    OPENCLAW_OPENAI_PROFILE,
    QWEN_CODE_OPENAI_PROFILE,
    detect_openai_client_profile,
    is_qwen_code_openai_request,
)


@dataclass(slots=True)
class StandardRequest:
    prompt: str
    response_model: str
    resolved_model: str
    surface: str
    client_profile: str = OPENCLAW_OPENAI_PROFILE
    requested_model: str | None = None
    content: str | None = None
    stream: bool = False
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    tool_name_registry: dict[str, str] = field(default_factory=dict)
    tool_enabled: bool = False
    attachments: list[NormalizedAttachment] = field(default_factory=list)
    uploaded_file_ids: list[str] = field(default_factory=list)
    upstream_files: list[dict[str, Any]] = field(default_factory=list)
    session_key: str | None = None
    context_mode: str = "inline"
    bound_account_email: str | None = None
    bound_account: Any | None = None
    stage_labels: dict[str, str] = field(default_factory=dict)
    full_prompt: str | None = None
    upstream_chat_id: str | None = None
    persistent_session: bool = False
    session_message_hashes: list[str] = field(default_factory=list)
    session_chat_invalidated: bool = False
