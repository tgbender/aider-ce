"""Token usage tracking and reporting utilities."""

import math
from typing import Any, Dict, List, Optional, Tuple


def extract_usage_from_completion(
    completion: Optional[Any],
    streaming_usage: Optional[Any] = None,
) -> Optional[Any]:
    """Extract usage data from completion or streaming usage.

    Providers may return usage in different ways:
    - Non-streaming: completion.usage contains all data
    - Streaming: final chunk often carries usage data

    This function prefers completion.usage but falls back to captured streaming usage.
    """
    if completion and hasattr(completion, "usage") and completion.usage is not None:
        return completion.usage
    elif streaming_usage is not None:
        return streaming_usage
    return None


def extract_cache_tokens(usage: Any) -> Tuple[int, int]:
    """Extract cache hit and write tokens from usage data.

    Handles multiple provider formats:
    - Anthropic: cache_read_input_tokens, cache_creation_input_tokens
    - DeepSeek: prompt_cache_hit_tokens
    - OpenAI: prompt_tokens_details.cached_tokens
    """
    cache_hit_tokens = 0
    cache_write_tokens = 0

    # Anthropic / DeepSeek style
    cache_hit_tokens = getattr(usage, "prompt_cache_hit_tokens", 0) or getattr(
        usage, "cache_read_input_tokens", 0
    )
    cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0)

    # OpenAI style: prompt_tokens_details.cached_tokens
    if not cache_hit_tokens:
        details = getattr(usage, "prompt_tokens_details", None)
        if isinstance(details, dict):
            cache_hit_tokens = details.get("cached_tokens", 0) or 0
        elif details is not None:
            cache_hit_tokens = getattr(details, "cached_tokens", 0) or 0

    return cache_hit_tokens or 0, cache_write_tokens or 0


def extract_thinking_tokens(usage: Any) -> int:
    """Extract thinking/reasoning tokens from usage data.

    Providers report thinking tokens in various locations:
    - output_tokens_details.reasoning_tokens
    - completion_tokens_details.reasoning_tokens
    - Flat usage.reasoning_tokens
    """
    thinking_tokens = 0

    # 1) Generic output_tokens_details.* (some providers)
    output_details = getattr(usage, "output_tokens_details", None)
    if isinstance(output_details, dict):
        thinking_tokens = (
            output_details.get("reasoning_tokens")
            or output_details.get("thinking_tokens")
            or 0
        )
    elif output_details is not None:
        thinking_tokens = (
            getattr(output_details, "reasoning_tokens", 0)
            or getattr(output_details, "thinking_tokens", 0)
            or 0
        )

    # 2) OpenAI-style: completion_tokens_details.reasoning_tokens
    if not thinking_tokens:
        completion_details = getattr(usage, "completion_tokens_details", None)
        if isinstance(completion_details, dict):
            thinking_tokens = (
                completion_details.get("reasoning_tokens")
                or completion_details.get("thinking_tokens")
                or 0
            )
        elif completion_details is not None:
            thinking_tokens = (
                getattr(completion_details, "reasoning_tokens", 0)
                or getattr(completion_details, "thinking_tokens", 0)
                or 0
            )

    # 3) Flat usage-level fields (fallback)
    if not thinking_tokens:
        thinking_tokens = (
            getattr(usage, "reasoning_tokens", 0)
            or getattr(usage, "thinking_tokens", 0)
            or 0
        )

    return thinking_tokens or 0


def detect_usage_format(usage: Any) -> Tuple[str, Dict[str, int]]:
    """Detect whether usage data is from raw Anthropic API or LiteLLM-normalized.

    Returns:
        Tuple of (format_type, token_counts)
        format_type: 'anthropic_raw', 'litellm_normalized', 'deepseek', or 'unknown'
        token_counts: dict with standardized field names
    """
    token_counts = {
        'prompt_tokens': getattr(usage, 'prompt_tokens', 0) or 0,
        'completion_tokens': getattr(usage, 'completion_tokens', 0) or 0,
        'cache_read_tokens': 0,
        'cache_write_tokens': 0,
    }

    # Check for raw Anthropic field names
    cache_read_input = getattr(usage, 'cache_read_input_tokens', None)
    cache_creation_input = getattr(usage, 'cache_creation_input_tokens', None)

    if cache_read_input is not None or cache_creation_input is not None:
        # Raw Anthropic format detected
        token_counts['cache_read_tokens'] = cache_read_input or 0
        token_counts['cache_write_tokens'] = cache_creation_input or 0

        # Check if prompt_tokens includes the cache tokens (LiteLLM normalized)
        total_cache = token_counts['cache_read_tokens'] + token_counts['cache_write_tokens']

        if total_cache > 0:
            # If prompt_tokens >= total_cache (with 10% tolerance), cache was likely folded in
            if token_counts['prompt_tokens'] >= total_cache * 0.9:
                return 'litellm_normalized', token_counts

        return 'anthropic_raw', token_counts

    # Check for LiteLLM/OpenAI style
    details = getattr(usage, 'prompt_tokens_details', None)
    if isinstance(details, dict) and details.get('cached_tokens'):
        token_counts['cache_read_tokens'] = details['cached_tokens']
        return 'litellm_normalized', token_counts
    elif details is not None:
        cached = getattr(details, 'cached_tokens', None)
        if cached:
            token_counts['cache_read_tokens'] = cached
            return 'litellm_normalized', token_counts

    # Check DeepSeek style
    prompt_cache_hit = getattr(usage, 'prompt_cache_hit_tokens', None)
    if prompt_cache_hit:
        token_counts['cache_read_tokens'] = prompt_cache_hit
        return 'deepseek', token_counts

    return 'unknown', token_counts


def validate_token_accounting(
    prompt_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    format_type: str,
) -> List[str]:
    """Validate token accounting and return list of warnings if something looks wrong."""
    warnings = []

    if format_type == 'anthropic_raw':
        # For raw Anthropic, prompt_tokens should be the transient tokens
        # So they should be <= total input (transient + cache_read + cache_write)
        total_input = prompt_tokens + cache_read_tokens + cache_write_tokens

        # Sanity check 1: transient should be non-negative
        if prompt_tokens < 0:
            warnings.append(f"Negative prompt_tokens detected: {prompt_tokens}")

        # Sanity check 2: Total should be reasonable (not zero, not absurdly high)
        if total_input == 0 and (prompt_tokens > 0 or cache_read_tokens > 0):
            warnings.append("Total input is 0 but component tokens are non-zero")

        # Note: For raw Anthropic, cache_read_tokens >> prompt_tokens is EXPECTED
        # with heavy caching (e.g., 200k cache, 50 transient). Don't warn here.

    elif format_type == 'litellm_normalized':
        # For LiteLLM, prompt_tokens should include cache hits
        # So: prompt_tokens >= cache_read_tokens (usually)
        if cache_read_tokens > prompt_tokens:
            warnings.append(
                f"Cache read ({cache_read_tokens}) exceeds prompt_tokens ({prompt_tokens}). "
                "This suggests raw Anthropic data was misidentified as LiteLLM-normalized."
            )

        # Check if subtraction would result in negative transient tokens
        transient = prompt_tokens - cache_read_tokens
        if transient < 0:
            warnings.append(
                f"Negative transient tokens calculated: {transient}. "
                "Token accounting mismatch!"
            )

    return warnings


def calculate_sent_tokens(
    prompt_tokens: int,
    cache_write_tokens: int,
    cache_hit_tokens: int,
    model_info: Dict[str, Any],
    model_name: str = "",
    usage_format: str = 'unknown',
) -> int:
    """Calculate total tokens sent for billing purposes.

    Different providers count sent tokens differently:
    - Anthropic: prompt_tokens excludes cache tokens, so add them
    - Others: prompt_tokens is already the full billed amount
    """
    provider = (model_info.get("litellm_provider") or "").lower()
    name_lower = model_name.lower()

    is_anthropic_provider = provider == "anthropic"
    is_bedrock_anthropic = provider == "bedrock" and (
        "anthropic" in name_lower or "claude" in name_lower
    )

    if is_anthropic_provider:
        # Native Anthropic: prompt_tokens excludes ALL cache tokens
        # Total sent = transient + cache_write + cache_read
        if usage_format == 'anthropic_raw':
            return prompt_tokens + cache_write_tokens + cache_hit_tokens
        else:
            # LiteLLM normalized: prompt_tokens already includes cache_read
            # We need to add cache_write since it's billed separately
            return prompt_tokens + cache_write_tokens
    else:
        # Other providers (including Bedrock/Vertex Anthropic):
        # prompt_tokens is already the full billed input
        return prompt_tokens


def format_token_count(count: int) -> str:
    """Format a token count for display (e.g., 15300 -> '15.3k')."""
    if count < 1000:
        return str(count)
    return f"{round(count / 1000)}k"


def format_token_report(
    message_tokens_sent: int,
    message_tokens_received: int,
    cache_write_tokens: int,
    cache_hit_tokens: int,
    completion_tokens: int,
    thinking_tokens: int,
) -> str:
    """Format a detailed token usage report string.

    Example output:
        "Tokens: 15.3k sent, 2.1k cache write, 8.2k cache read; 4.5k Total (1.2k thinking, 3.3k returned)"
    """
    tokens_report = f"Tokens: {format_token_count(message_tokens_sent)} sent"

    if cache_write_tokens:
        tokens_report += f", {format_token_count(cache_write_tokens)} cache write"
    if cache_hit_tokens:
        tokens_report += f", {format_token_count(cache_hit_tokens)} cache read"

    # Show completion breakdown with thinking vs returned
    if completion_tokens:
        total_str = format_token_count(completion_tokens)
        if thinking_tokens and completion_tokens >= thinking_tokens:
            returned_tokens = completion_tokens - thinking_tokens
            tokens_report += (
                f"; {total_str} Total "
                f"({format_token_count(thinking_tokens)} thinking, "
                f"{format_token_count(returned_tokens)} returned)"
            )
        else:
            tokens_report += f"; {total_str} Total"

    return tokens_report


def format_cost(value: float) -> str:
    """Format a cost value for display.

    Shows 2 decimal places for values >= 0.01, more precision for smaller values.
    """
    if value == 0:
        return "0.00"
    magnitude = abs(value)
    if magnitude >= 0.01:
        return f"{value:.2f}"
    else:
        precision = max(2, 2 - int(math.log10(magnitude)))
        return f"{value:.{precision}f}"


def format_cost_report(
    message_cost: float,
    total_cost: float,
    cost_breakdown: Dict[str, float],
) -> str:
    """Format a detailed cost report string with breakdown.

    Example output:
        "Cost: $0.0423 message (input $0.0150, output $0.0225, cache read $0.0032, cache write $0.0016), $1.2345 session."
    """
    breakdown_parts = [
        f"input ${format_cost(cost_breakdown.get('input_cost', 0))}",
        f"output ${format_cost(cost_breakdown.get('output_cost', 0))}",
    ]

    cache_read_cost = cost_breakdown.get("cache_read_cost", 0)
    cache_write_cost = cost_breakdown.get("cache_write_cost", 0)

    if cache_read_cost:
        breakdown_parts.append(f"cache read ${format_cost(cache_read_cost)}")
    if cache_write_cost:
        breakdown_parts.append(f"cache write ${format_cost(cache_write_cost)}")

    breakdown_str = ", ".join(breakdown_parts)

    cost_report = (
        f"Cost: ${format_cost(message_cost)} message"
        f" ({breakdown_str}),"
        f" ${format_cost(total_cost)} session."
    )

    return cost_report


def calculate_message_costs(
    usage: Any,
    model_info: Dict[str, Any],
    model_name: str,
    partial_response_content: str,
    messages: list,
    token_counter: callable,
) -> Dict[str, Any]:
    """Calculate all message costs and return a comprehensive result.

    This is the main entry point for calculating costs from usage data.
    Handles fallback to token counting when usage is unavailable.

    Returns a dictionary with:
    - prompt_tokens: Input tokens
    - completion_tokens: Output tokens
    - cache_hit_tokens: Cache read tokens
    - cache_write_tokens: Cache write tokens
    - thinking_tokens: Thinking/reasoning tokens
    - returned_tokens: Non-thinking output tokens
    - message_tokens_sent: Total tokens sent (for billing)
    - message_tokens_received: Total tokens received
    - _usage_format: Detected format type (for debugging)
    - _validation_warnings: Any validation warnings (for debugging)
    """
    from .cost_calculator import calculate_cost_breakdown

    # Detect the usage data format
    if usage is not None:
        usage_format, token_counts = detect_usage_format(usage)
        prompt_tokens = token_counts['prompt_tokens']
        completion_tokens = token_counts['completion_tokens']
        cache_hit_tokens = token_counts['cache_read_tokens']
        cache_write_tokens = token_counts['cache_write_tokens']
        thinking_tokens = extract_thinking_tokens(usage)

        # Validate token accounting and capture warnings
        validation_warnings = validate_token_accounting(
            prompt_tokens, cache_hit_tokens, cache_write_tokens, usage_format
        )
    else:
        # Fallback: estimate from token counter
        usage_format = 'unknown'
        prompt_tokens = token_counter(messages)
        completion_tokens = token_counter(partial_response_content)
        cache_hit_tokens = 0
        cache_write_tokens = 0
        thinking_tokens = 0
        validation_warnings = []

    # Calculate returned tokens (output minus thinking)
    if thinking_tokens and completion_tokens and completion_tokens >= thinking_tokens:
        returned_tokens = completion_tokens - thinking_tokens
    else:
        thinking_tokens = 0
        returned_tokens = completion_tokens

    # Calculate tokens sent for billing with format awareness
    message_tokens_sent = calculate_sent_tokens(
        prompt_tokens, cache_write_tokens, cache_hit_tokens,
        model_info, model_name, usage_format
    )
    message_tokens_received = completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_write_tokens": cache_write_tokens,
        "thinking_tokens": thinking_tokens,
        "returned_tokens": returned_tokens,
        "message_tokens_sent": message_tokens_sent,
        "message_tokens_received": message_tokens_received,
        "_usage_format": usage_format,
        "_validation_warnings": validation_warnings,
    }
