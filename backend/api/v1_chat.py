from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import logging
import uuid
from backend.services.qwen_client import QwenClient
from backend.services.token_calc import calculate_usage
from backend.services.prompt_builder import build_prompt_with_tools

log = logging.getLogger("qwen2api.chat")
router = APIRouter()

@router.post("/completions")
@router.post("/chat/completions")
@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    app = request.app
    users_db = app.state.users_db
    client: QwenClient = app.state.qwen_client

    # 鉴权 (完全复原单文件逻辑)
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""

    if not token:
        token = request.headers.get("x-api-key", "").strip()
    if not token:
        token = request.query_params.get("key", "").strip() or request.query_params.get("api_key", "").strip()

    from backend.core.config import API_KEYS, settings
    admin_k = settings.ADMIN_KEY

    # 兼容处理逻辑：
    # 1. 没有配置 API_KEYS 则默认放行
    # 2. 若配置了，则接受 admin_key 或存在于 API_KEYS 中的 key
    # 3. 甚至接受任何非空 key（放宽限制，以支持各种三方工具自带 key）
    if API_KEYS:
        if token != admin_k and token not in API_KEYS and not token:
            raise HTTPException(status_code=401, detail="Invalid API Key")

    # 获取下游用户并处理配额（如果该功能启用且存在对应的用户）
    users = await users_db.get()
    user = next((u for u in users if u["id"] == token), None)
    if user and user.get("quota", 0) <= user.get("used_tokens", 0):
        raise HTTPException(status_code=402, detail="Quota Exceeded")
        
    body = await request.json()
    from backend.core.config import resolve_model
    model = resolve_model(body.get("model", "gpt-3.5-turbo"))
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    
    # 构建带指令劫持的 Prompt
    content = build_prompt_with_tools(messages, tools)
    
    log.info(f"[OAI] model={model}, stream=True, tools={[t.get('function', {}).get('name') for t in tools]}, prompt_len={len(content)}")

    # 无感重试调用
    async def generate():
        current_prompt = content
        
        for stream_attempt in range(5):
            try:
                events, chat_id, acc = await client.chat_stream_events_with_retry(model, current_prompt)
                
                # Buffer all events
                thinking_chunks = []
                answer_chunks = []
                for evt in events:
                    if evt.get("type") != "delta":
                        continue
                    phase = evt.get("phase", "")
                    cont = evt.get("content", "")
                    if phase in ("think", "thinking_summary") and cont:
                        thinking_chunks.append(cont)
                    elif phase == "answer" and cont:
                        answer_chunks.append(cont)
                    if evt.get("status") == "finished" and phase == "answer":
                        break
                        
                answer_text = "".join(answer_chunks)
                reasoning_text = "".join(thinking_chunks)
                
                # Detect Qwen native tool call interception
                import re
                native_blocked_m = re.match(r'^Tool (\w+) does not exists?\.?$', answer_text.strip())
                if native_blocked_m and tools and stream_attempt < 4:
                    blocked_name = native_blocked_m.group(1)
                    client.account_pool.release(acc)
                    import asyncio
                    asyncio.create_task(client.delete_chat(acc.token, chat_id))
                    log.warning(f"[NativeBlock-OAI] Qwen拦截原生工具调用 '{blocked_name}'，重试 (attempt {stream_attempt+1}/5)")
                    from backend.services.tool_parser import inject_format_reminder
                    current_prompt = inject_format_reminder(current_prompt, blocked_name)
                    await asyncio.sleep(0.5)
                    continue
                    
                # Parse tools
                from backend.services.tool_parser import parse_tool_calls
                if tools:
                    blocks, stop_reason = parse_tool_calls(answer_text, tools)
                else:
                    blocks = [{"type": "text", "text": answer_text}]
                    stop_reason = "stop"
                    
                if stop_reason == "end_turn":
                    stop_reason = "stop"
                    
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                
                # Thinking blocks
                if reasoning_text:
                    # In OpenAI format, reasoning is usually not natively supported in chunks unless using reasoning_content,
                    # but we can output it as content or ignore it. For now, we'll output it as content.
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": reasoning_text}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    
                for blk in blocks:
                    if blk["type"] == "text" and blk.get("text"):
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": blk["text"]}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif blk["type"] == "tool_use":
                        tc_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [{"index": 0, "delta": {
                                "tool_calls": [{
                                    "id": blk["id"],
                                    "type": "function",
                                    "function": {
                                        "name": blk["name"],
                                        "arguments": json.dumps(blk.get("input", {}), ensure_ascii=False)
                                    }
                                }]
                            }, "finish_reason": "tool_calls"}]
                        }
                        yield f"data: {json.dumps(tc_chunk)}\n\n"
                        
                # Final chunk
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": stop_reason}]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                
                log.info(f"[OAI] Request complete. Generated {len(answer_text)} characters.")
                
                users = await users_db.get()
                for u in users:
                    if u["id"] == token:
                        u["used_tokens"] += len(answer_text) + len(current_prompt)
                        break
                await users_db.save(users)
                
                client.account_pool.release(acc)
                import asyncio
                asyncio.create_task(client.delete_chat(acc.token, chat_id))
                return
                
            except Exception as e:
                log.error(f"Chat request failed: {e}")
                return

    return StreamingResponse(generate(), media_type="text/event-stream")
