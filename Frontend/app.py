from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import httpx
import json
import asyncio
import time
import logging
from datetime import datetime
from typing import AsyncGenerator
import os
from dotenv import load_dotenv

load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['BACKEND_URL'] = os.getenv('BACKEND_URL', 'http://localhost:8000')
app.config['LANGUAGE'] = os.getenv('LANGUAGE', 'ja')


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


app.config['ENABLE_DI_LINK'] = _env_bool('ENABLE_DI_LINK', False)

# Session storage (use Redis in production)
messages_store = {}


FRONT_TEXT = {
    "ja": {
        "log_front_request_received": "📨 フロントエンド: リクエスト受信 (model={model})",
        "log_front_send_backend": "🔗 バックエンドへリクエスト送信 (model={model})",
        "log_front_first_chunk": "⏱️ 最初のチャンク受信 (待機時間: {ms}ms)",
        "log_front_completed_chunks": "✅ フロントエンド完了 (総時間: {s}s, チャンク数: {count})",
        "log_front_guideline_request": "📨 RAG検索リクエスト受信 (model={model})",
        "log_front_multi_request_received": "📨 フロントエンド: マルチエージェントリクエスト受信 (model={model})",
        "log_front_send_multi_backend": "🔗 バックエンドへマルチエージェントリクエスト送信 (model={model})",
        "log_front_first_response": "⏱️ 最初のレスポンス受信 (待機時間: {ms}ms)",
        "log_front_completed_lines": "✅ フロントエンド完了 (総時間: {s}s, ライン数: {count})",
        "log_front_board_request_received": "📨 フロントエンド: 井戸端会議リクエスト受信 (model={model}, tone={tone})",
        "log_front_send_phase1_backend": "🔗 バックエンドへフェーズ1リクエスト送信 (model={model}, tone={tone})",
        "label_multi_agent": "🔀 [マルチエージェント分析]",
        "label_idobata": "🗣️ [井戸端会議]",
        "error_block": "\n\n❌ エラー: {error}",
        "error_inline": "❌ エラー: {error}",
    },
    "en": {
        "log_front_request_received": "📨 Frontend: Request received (model={model})",
        "log_front_send_backend": "🔗 Sending request to backend (model={model})",
        "log_front_first_chunk": "⏱️ First chunk received (wait time: {ms}ms)",
        "log_front_completed_chunks": "✅ Frontend completed (total time: {s}s, chunks: {count})",
        "log_front_guideline_request": "📨 RAG search request received (model={model})",
        "log_front_multi_request_received": "📨 Frontend: Multi-agent request received (model={model})",
        "log_front_send_multi_backend": "🔗 Sending multi-agent request to backend (model={model})",
        "log_front_first_response": "⏱️ First response received (wait time: {ms}ms)",
        "log_front_completed_lines": "✅ Frontend completed (total time: {s}s, lines: {count})",
        "log_front_board_request_received": "📨 Frontend: AI board meeting request received (model={model}, tone={tone})",
        "log_front_send_phase1_backend": "🔗 Sending phase 1 request to backend (model={model}, tone={tone})",
        "label_multi_agent": "🔀 [Multi-Agent Analysis]",
        "label_idobata": "🗣️ [AI Board Meeting]",
        "error_block": "\n\n❌ Error: {error}",
        "error_inline": "❌ Error: {error}",
    },
}


def front_text(key: str, **kwargs) -> str:
    """Return a localized frontend text for the configured LANGUAGE."""
    lang = app.config.get("LANGUAGE", "ja")
    table = FRONT_TEXT.get(lang) or FRONT_TEXT["ja"]
    template = table.get(key) or FRONT_TEXT["ja"].get(key) or key
    return template.format(**kwargs)


@app.route('/')
def index():
    """Main page"""
    language = app.config['LANGUAGE']
    enable_di_link = app.config.get('ENABLE_DI_LINK', False)
    return render_template('index.html', language=language, enable_di_link=enable_di_link)


@app.route('/api/messages', methods=['GET'])
def get_messages():
    """Get message history"""
    session_id = request.args.get('session_id', 'default')
    messages = messages_store.get(session_id, [])
    return jsonify(messages)


@app.route('/api/messages/clear', methods=['POST'])
def clear_messages():
    """Clear message history"""
    data = request.json
    session_id = data.get('session_id', 'default')
    messages_store[session_id] = []
    return jsonify({'status': 'ok'})


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Regular chat streaming"""
    start_time = time.time()
    request_id = f"front_req_{int(start_time * 1000)}"
    
    data = request.json
    prompt = data.get('prompt', '')
    session_id = data.get('session_id', 'default')
    model = data.get('model', 'gpt-4.1-mini')
    
    logger.info(f"[{request_id}] {front_text('log_front_request_received', model=model)}")
    
    if not prompt:
        return jsonify({'error': 'prompt required'}), 400
    
    # Save user message
    if session_id not in messages_store:
        messages_store[session_id] = []
    
    user_message = {
        'is_user': True,
        'content': prompt,
        'timestamp': datetime.now().isoformat()
    }
    messages_store[session_id].append(user_message)
    
    def generate():
        """Generate streaming response"""
        ai_content = ""
        try:
            backend_request_start = time.time()
            logger.info(f"[{request_id}] {front_text('log_front_send_backend', model=model)}")
            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    'POST',
                    f"{app.config['BACKEND_URL']}/api/stream",
                    json={'prompt': prompt, 'model': model}
                ) as response:
                    response.raise_for_status()
                    first_chunk = True
                    chunk_count = 0
                    for chunk in response.iter_text():
                        if chunk:
                            if first_chunk:
                                logger.info(
                                    f"[{request_id}] {front_text('log_front_first_chunk', ms=f'{(time.time() - backend_request_start)*1000:.2f}') }"
                                )
                                first_chunk = False
                            chunk_count += 1
                            ai_content += chunk
                            yield chunk
            
            total_time = time.time() - start_time
            logger.info(
                f"[{request_id}] {front_text('log_front_completed_chunks', s=f'{total_time:.2f}', count=chunk_count)}"
            )
            
            # Save AI message
            ai_message = {
                'is_user': False,
                'content': ai_content,
                'timestamp': datetime.now().isoformat(),
                'is_streaming': False
            }
            messages_store[session_id].append(ai_message)
            
        except Exception as e:
            error_msg = front_text('error_block', error=str(e))
            yield error_msg
            
            ai_message = {
                'is_user': False,
                'content': ai_content + error_msg,
                'timestamp': datetime.now().isoformat(),
                'is_streaming': False
            }
            messages_store[session_id].append(ai_message)
    
    return Response(stream_with_context(generate()), content_type='text/plain')


@app.route('/api/rag/stream', methods=['POST'])
@app.route('/api/guideline/stream', methods=['POST'])
def guideline_stream():
    """RAG search chat streaming (backward compatible with /api/guideline/stream)"""
    data = request.json
    prompt = data.get('prompt', '')
    model = data.get('model', 'gpt-4.1-mini')

    logger.info(front_text('log_front_guideline_request', model=model))
    
    if not prompt:
        return jsonify({'error': 'prompt required'}), 400
    
    def generate():
        """Generate streaming response"""
        try:
            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    'POST',
                    f"{app.config['BACKEND_URL']}/api/rag/stream",
                    json={'prompt': prompt, 'model': model}
                ) as response:
                    response.raise_for_status()
                    for chunk in response.iter_text():
                        if chunk:
                            yield chunk
            
        except Exception as e:
            error_msg = front_text('error_block', error=str(e))
            yield error_msg
    
    return Response(stream_with_context(generate()), content_type='text/plain')


@app.route('/api/chat/multi-agent-stream', methods=['POST'])
def multi_agent_stream():
    """Multi-agent chat streaming"""
    start_time = time.time()
    request_id = f"front_multi_{int(start_time * 1000)}"
    
    data = request.json
    prompt = data.get('prompt', '')
    session_id = data.get('session_id', 'default')
    model = data.get('model', 'gpt-4.1-mini')
    
    logger.info(f"[{request_id}] {front_text('log_front_multi_request_received', model=model)}")
    
    if not prompt:
        return jsonify({'error': 'prompt required'}), 400
    
    # Save user message
    if session_id not in messages_store:
        messages_store[session_id] = []
    
    user_message = {
        'is_user': True,
        'content': f"{front_text('label_multi_agent')} {prompt}",
        'timestamp': datetime.now().isoformat()
    }
    messages_store[session_id].append(user_message)
    
    def generate():
        """Generate multi-agent streaming response"""
        ai_message = {
            'is_user': False,
            'is_multi_agent': True,
            'timestamp': datetime.now().isoformat(),
            'critical_content': '',
            'positive_content': '',
            'synthesis_content': ''
        }
        
        try:
            backend_request_start = time.time()
            logger.info(f"[{request_id}] {front_text('log_front_send_multi_backend', model=model)}")
            first_response = True
            line_count = 0
            with httpx.Client(timeout=120.0) as client:
                with client.stream(
                    'POST',
                    f"{app.config['BACKEND_URL']}/api/multi-agent-stream",
                    json={'prompt': prompt, 'model': model}
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.strip():
                            try:
                                if first_response:
                                    logger.info(
                                        f"[{request_id}] {front_text('log_front_first_response', ms=f'{(time.time() - backend_request_start)*1000:.2f}') }"
                                    )
                                    first_response = False
                                line_count += 1
                                # Send in JSON-Lines format
                                yield line + '\n'
                                
                                # Update message store
                                data = json.loads(line)
                                if 'agent' in data and 'content' in data:
                                    agent = data['agent']
                                    content = data['content']
                                    
                                    if agent == 'CriticalAnalyst':
                                        ai_message['critical_content'] += content
                                    elif agent == 'PositiveAdvocate':
                                        ai_message['positive_content'] += content
                                    elif agent == 'Synthesizer':
                                        ai_message['synthesis_content'] += content
                            except json.JSONDecodeError:
                                continue
            
            total_time = time.time() - start_time
            logger.info(
                f"[{request_id}] {front_text('log_front_completed_lines', s=f'{total_time:.2f}', count=line_count)}"
            )
            
            # Save AI message
            messages_store[session_id].append(ai_message)
            
        except Exception as e:
            error_data = {'type': 'error', 'message': str(e)}
            yield json.dumps(error_data) + '\n'
            
            ai_message['synthesis_content'] = front_text('error_inline', error=str(e))
            messages_store[session_id].append(ai_message)
    
    return Response(stream_with_context(generate()), content_type='text/plain')


@app.route('/api/chat/idobata-stream', methods=['POST'])
def idobata_stream():
    """AI Board Meeting (Phase 1 Planning) streaming"""
    start_time = time.time()
    request_id = f"front_idobata_{int(start_time * 1000)}"

    data = request.json
    prompt = data.get('prompt', '')
    session_id = data.get('session_id', 'idobata')
    model = data.get('model', 'gpt-4.1-mini')
    tone = data.get('tone', 'balanced')

    logger.info(
        f"[{request_id}] {front_text('log_front_board_request_received', model=model, tone=tone)}"
    )

    if not prompt:
        return jsonify({'error': 'prompt required'}), 400

    if session_id not in messages_store:
        messages_store[session_id] = []

    user_message = {
        'is_user': True,
        'content': f"{front_text('label_idobata')} {prompt}",
        'timestamp': datetime.now().isoformat()
    }
    messages_store[session_id].append(user_message)

    def generate():
        ai_message = {
            'is_user': False,
            'is_planning': True,
            'timestamp': datetime.now().isoformat(),
            'planning_content': '',
            'tech_content': '',
            'business_content': '',
            'synthesis_content': ''
        }

        try:
            backend_request_start = time.time()
            logger.info(
                f"[{request_id}] {front_text('log_front_send_phase1_backend', model=model, tone=tone)}"
            )
            first_response = True
            line_count = 0
            with httpx.Client(timeout=180.0) as client:
                with client.stream(
                    'POST',
                    f"{app.config['BACKEND_URL']}/api/phase1/stream",
                    json={'prompt': prompt, 'model': model, 'tone': tone}
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.strip():
                            try:
                                if first_response:
                                    logger.info(
                                        f"[{request_id}] {front_text('log_front_first_response', ms=f'{(time.time() - backend_request_start)*1000:.2f}') }"
                                    )
                                    first_response = False
                                line_count += 1
                                yield line + '\n'

                                data = json.loads(line)
                                if 'agent' in data and 'content' in data:
                                    agent = data['agent']
                                    content = data['content']

                                    # Backend (/api/phase1/stream) returns CEO/CTO/CFO/COO
                                    if agent == 'CEO':
                                        ai_message['planning_content'] += content
                                    elif agent == 'CTO':
                                        ai_message['tech_content'] += content
                                    elif agent == 'CFO':
                                        ai_message['business_content'] += content
                                    elif agent == 'COO':
                                        ai_message['synthesis_content'] += content
                            except json.JSONDecodeError:
                                continue

            total_time = time.time() - start_time
            logger.info(
                f"[{request_id}] {front_text('log_front_completed_lines', s=f'{total_time:.2f}', count=line_count)}"
            )

            messages_store[session_id].append(ai_message)

        except Exception as e:
            error_data = {'type': 'error', 'message': str(e)}
            yield json.dumps(error_data) + '\n'

            ai_message['synthesis_content'] = front_text('error_inline', error=str(e))
            messages_store[session_id].append(ai_message)

    return Response(stream_with_context(generate()), content_type='text/plain')


if __name__ == '__main__':
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes", "on")
    host = os.getenv("HOST", "0.0.0.0")
    app.run(debug=debug, host=host, port=port, threaded=True)
