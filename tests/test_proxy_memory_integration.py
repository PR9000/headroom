"""Integration tests for proxy memory system with real API calls.

These tests require:
- ANTHROPIC_API_KEY environment variable set

Run with:
    ANTHROPIC_API_KEY=... uv run pytest tests/test_proxy_memory_integration.py -v

Test categories:
- TestMemoryHeaderValidation: User ID header validation
- TestMemoryToolInjection: Memory tools are injected
- TestMemorySaveAndSearch: End-to-end save/recall flow
- TestMemoryUserIsolation: User memory isolation
"""

import os
import tempfile
import time
from pathlib import Path

import pytest

# Set tokenizer parallelism before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


@pytest.fixture
def temp_memory_db():
    """Create temporary memory database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)
    # Also cleanup related files (HNSW index, etc.)
    for suffix in ["-shm", "-wal", ".hnsw"]:
        Path(f.name + suffix).unlink(missing_ok=True)


@pytest.fixture
def memory_client(temp_memory_db):
    """Create test client with memory enabled."""
    config = ProxyConfig(
        optimize=False,  # Disable optimization for simpler tests
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        memory_enabled=True,
        memory_backend="local",
        memory_db_path=temp_memory_db,
        memory_inject_tools=True,
        memory_inject_context=True,
        memory_top_k=5,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def no_memory_client():
    """Create test client with memory disabled."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        memory_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key from environment."""
    return os.environ.get("ANTHROPIC_API_KEY")


class TestMemoryHeaderValidation:
    """Test user ID header validation."""

    def test_missing_user_id_uses_default(self, memory_client, anthropic_api_key):
        """Request without x-headroom-user-id should use 'default' user for simple DevEx."""
        if not anthropic_api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                # Note: NOT setting x-headroom-user-id - should default to "default"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        # Should succeed, not return 400
        assert response.status_code == 200

    def test_with_user_id_succeeds(self, memory_client, anthropic_api_key):
        """Request with x-headroom-user-id should succeed."""
        if not anthropic_api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": "test-user-123",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello, just say hi back."}],
            },
        )
        assert response.status_code == 200

    def test_no_memory_client_doesnt_require_user_id(self, no_memory_client, anthropic_api_key):
        """When memory is disabled, user ID header should not be required."""
        if not anthropic_api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        response = no_memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                # No x-headroom-user-id
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello, just say hi."}],
            },
        )
        assert response.status_code == 200


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestMemoryToolInjection:
    """Test memory tool injection."""

    def test_memory_tools_are_available(self, memory_client, anthropic_api_key):
        """Memory tools should be available to the LLM."""
        response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": "test-user-tool-check",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "user",
                        "content": "List the tools available to you. Just list the tool names.",
                    }
                ],
            },
        )
        assert response.status_code == 200

        # The response should mention memory tools
        content = response.json().get("content", [])
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")

        # At least one memory tool should be mentioned
        assert any(tool in text.lower() for tool in ["memory_save", "memory_search", "memory"]), (
            f"Memory tools not found in response: {text}"
        )


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestMemorySaveAndSearch:
    """Test memory save and search flow."""

    def test_save_memory_via_explicit_instruction(self, memory_client, anthropic_api_key):
        """LLM should be able to save memories when instructed."""
        user_id = f"test-user-save-{int(time.time())}"

        # Request that explicitly asks to save
        response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": user_id,
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": "Please save this to memory: My favorite programming language is Rust. "
                        "Use the memory_save tool to save this information.",
                    }
                ],
            },
        )
        assert response.status_code == 200

        # Check if response indicates tool was used
        resp_json = response.json()
        content = resp_json.get("content", [])

        # Response could be tool_use (if not handled) or text (if handled)
        # Either way, it should complete successfully
        assert content, "Response should have content"

    def test_save_and_recall_memory(self, memory_client, anthropic_api_key):
        """Save a memory and recall it in subsequent request."""
        user_id = f"test-user-recall-{int(time.time())}"

        # First request: save a memory with explicit instruction
        save_response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": user_id,
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": "Please remember this: My name is TestUser and I work at AcmeCorp. "
                        "Save this information using the memory_save tool.",
                    }
                ],
            },
        )
        assert save_response.status_code == 200

        # Wait a moment for memory to be indexed
        time.sleep(1)

        # Second request: ask about saved info
        # Memory context should be injected automatically
        recall_response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": user_id,
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 300,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is my name and where do I work? "
                        "Answer based on what you know about me.",
                    }
                ],
            },
        )
        assert recall_response.status_code == 200

        # Check if response mentions the saved info
        content = recall_response.json().get("content", [])
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")

        # Should mention at least one of the saved facts
        text_lower = text.lower()
        assert "testuser" in text_lower or "acmecorp" in text_lower or "acme" in text_lower, (
            f"Saved info not recalled: {text}"
        )


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestMemoryUserIsolation:
    """Test that memories are isolated per user."""

    def test_different_users_have_isolated_memories(self, memory_client, anthropic_api_key):
        """User A's memories should not appear for User B."""
        timestamp = int(time.time())
        user_a = f"user-a-isolation-{timestamp}"
        user_b = f"user-b-isolation-{timestamp}"
        secret_code = f"SECRETCODE{timestamp}"

        # Save memory for user A
        save_response = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": user_a,
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Remember my secret code: {secret_code}. "
                        "Save this using the memory_save tool.",
                    }
                ],
            },
        )
        assert save_response.status_code == 200

        # Wait for memory to be indexed
        time.sleep(1)

        # Query as user B - should NOT have access to user A's memory
        response_b = memory_client.post(
            "/v1/messages",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": user_b,
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 300,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is my secret code? Search your memory for it.",
                    }
                ],
            },
        )
        assert response_b.status_code == 200

        # User B should NOT see user A's secret code
        content = response_b.json().get("content", [])
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")

        assert secret_code not in text, f"User B should not see User A's secret: {text}"


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestMemoryStats:
    """Test memory-related stats and health."""

    def test_health_endpoint_works_with_memory(self, memory_client):
        """Health endpoint should work when memory is enabled."""
        response = memory_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"

    def test_stats_endpoint_works_with_memory(self, memory_client):
        """Stats endpoint should work when memory is enabled."""
        response = memory_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "requests" in data


@pytest.fixture
def memory_client_global(temp_memory_db):
    """Memory-enabled client with GLOBAL storage mode.

    GLOBAL keeps every memory in a single SQLite file regardless of
    project routing, so tests that pre-seed via direct backend access
    are guaranteed to share the same DB as the proxy's runtime backend.
    """
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        memory_enabled=True,
        memory_backend="local",
        memory_db_path=temp_memory_db,
        memory_inject_tools=True,
        memory_inject_context=True,
        memory_top_k=5,
        memory_storage_mode="global",
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestMemoryIdAutoTailAndUpdate:
    """End-to-end live: model uses [memory_id] from auto-tail to call memory_update.

    Validates that the IDs we added to the auto-injected memory block
    (see ``MemoryHandler.search_and_format_context``) are extractable by
    a real Claude model and can be passed directly to ``memory_update``
    without an intervening ``memory_search`` round-trip.
    """

    def test_model_uses_memory_id_to_call_memory_update(
        self,
        memory_client_global,
        anthropic_api_key,
        temp_memory_db,
    ):
        import asyncio

        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

        user_id = f"test-id-update-{int(time.time())}"

        # Pre-seed a known memory directly via a fresh backend so we
        # control its content and learn its ID up front. Same db_path
        # AND same embedder (ONNX) as the proxy backend → both read
        # the same SQLite file and produce comparable vectors.
        async def _seed() -> str:
            backend = LocalBackend(
                LocalBackendConfig(
                    db_path=temp_memory_db,
                    embedder_backend="onnx",
                    embedder_model="all-MiniLM-L6-v2",
                    vector_dimension=384,
                )
            )
            mem = await backend.save_memory(
                content="The user's favorite color is blue.",
                user_id=user_id,
            )
            return mem.id

        memory_id = asyncio.run(_seed())
        assert memory_id, "save_memory should return a usable id"

        # Let the SQLite write + index settle before the proxy reads.
        time.sleep(0.5)

        # Capture tool calls the proxy executes, so we can assert the
        # model picked the right tool with the right memory_id.
        proxy = memory_client_global.app.state.proxy
        assert proxy.memory_handler is not None
        recorded: list[dict] = []
        original_execute = proxy.memory_handler._execute_memory_tool

        async def _capturing_execute(
            tool_name, input_data, user_id_arg, provider, request_context=None
        ):
            recorded.append({"tool_name": tool_name, "input": dict(input_data)})
            return await original_execute(
                tool_name,
                input_data,
                user_id_arg,
                provider,
                request_context=request_context,
            )

        proxy.memory_handler._execute_memory_tool = _capturing_execute  # type: ignore[assignment]

        try:
            response = memory_client_global.post(
                "/v1/messages",
                headers={
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "x-headroom-user-id": user_id,
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 800,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Quick correction: my favorite color is actually "
                                "green, not blue. Please call the memory_update "
                                "tool to fix the relevant memory in your context. "
                                "The relevant memories block lists each memory's "
                                "ID in square brackets — use that ID for "
                                "memory_id."
                            ),
                        }
                    ],
                },
            )
        finally:
            proxy.memory_handler._execute_memory_tool = original_execute  # type: ignore[assignment]

        assert response.status_code == 200, response.text

        # The model should have called memory_update at least once.
        update_calls = [c for c in recorded if c["tool_name"] == "memory_update"]
        assert update_calls, f"Expected at least one memory_update call. Recorded: {recorded}"

        # And it should reference the exact ID we seeded — i.e. the
        # model used the [id] from the auto-tail block, not a guess.
        assert any(c["input"].get("memory_id") == memory_id for c in update_calls), (
            f"Expected memory_update(memory_id={memory_id!r}); got inputs: "
            f"{[c['input'] for c in update_calls]}"
        )
