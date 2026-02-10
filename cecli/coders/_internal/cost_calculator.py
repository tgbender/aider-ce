"""Cost calculation utilities with provider-aware pricing logic."""

from typing import Dict, Any, Tuple


def detect_provider_style(model_info: Dict[str, Any], model_name: str = "") -> Tuple[bool, bool]:
    """Detect the provider type and cache semantics.

    Returns:
        Tuple of (is_anthropic_provider, is_bedrock_anthropic)
        - is_anthropic_provider: Native Anthropic API (excludes cache tokens from prompt_tokens)
        - is_bedrock_anthropic: Bedrock/Vertex with Anthropic models (includes cache tokens)
    """
    provider = (model_info.get("litellm_provider") or "").lower()
    name_lower = model_name.lower()

    is_anthropic_provider = provider == "anthropic"
    is_bedrock_anthropic = provider == "bedrock" and (
        "anthropic" in name_lower or "claude" in name_lower
    )

    return is_anthropic_provider, is_bedrock_anthropic


def get_cache_rates(model_info: Dict[str, Any]) -> Tuple[float, float]:
    """Extract cache read and write cost rates from model info.

    Returns:
        Tuple of (cache_read_rate, cache_write_rate)
    """
    input_cost = model_info.get("input_cost_per_token") or 0

    # Cache read rate - try multiple field names
    cache_read_rate = (
        model_info.get("cache_read_input_token_cost")
        or model_info.get("input_cost_per_token_cache_hit")
        or 0
    )

    # Cache write rate
    cache_write_rate = model_info.get("cache_creation_input_token_cost") or 0

    return cache_read_rate, cache_write_rate


def calculate_input_cost(
    prompt_tokens: int,
    cache_hit_tokens: int,
    cache_write_tokens: int,
    input_cost_per_token: float,
    cache_read_rate: float,
    cache_write_rate: float,
    is_anthropic_style: bool,
    provider: str,
    model_name: str,
    usage_format: str = 'unknown',
) -> Tuple[float, float, float, float]:
    """Calculate input costs with provider-aware cache handling.

    For Anthropic-style providers (native Anthropic API):
    - prompt_tokens IS the transient tokens (excludes cache read/write)
    - Total input = prompt_tokens + cache_hit_tokens + cache_write_tokens
    - Bill transient tokens at input rate
    - Bill cache reads and writes at their respective rates

    For other providers (OpenAI, Azure, Bedrock Anthropic, DeepSeek):
    - prompt_tokens includes cache hits
    - If cache read rate known, bill hits at that rate, rest at input rate
    - If no cache read rate, bill everything at input rate
    - Cache writes use explicit rate or 1.25x heuristic

    Args:
        usage_format: 'anthropic_raw', 'litellm_normalized', or 'unknown'
                     Determines how to interpret prompt_tokens

    Returns:
        Tuple of (input_cost, cache_read_cost, cache_write_cost, transient_tokens)
    """
    input_cost = 0.0
    cache_read_cost = 0.0
    cache_write_cost = 0.0
    transient_tokens = 0

    if is_anthropic_style:
        # NATIVE ANTHROPIC API SEMANTICS
        # For raw Anthropic: prompt_tokens IS the transient tokens (already excludes cache)
        # For LiteLLM-normalized: prompt_tokens may include cache tokens

        # Determine if LiteLLM normalized the data by checking if prompt_tokens includes cache
        total_cache = cache_hit_tokens + cache_write_tokens

        if usage_format == 'litellm_normalized' or (
            total_cache > 0 and prompt_tokens >= total_cache * 0.9
        ):
            # LiteLLM included cache in prompt_tokens, subtract to get transient
            transient_tokens = max(0, prompt_tokens - cache_hit_tokens - cache_write_tokens)
        else:
            # Raw Anthropic: prompt_tokens is already just transient tokens
            transient_tokens = prompt_tokens

        # Bill transient tokens at full input rate
        input_cost = transient_tokens * input_cost_per_token

        # Bill cache reads (typically 10% of input cost for Anthropic)
        if cache_hit_tokens:
            read_rate = cache_read_rate or (input_cost_per_token * 0.1)
            cache_read_cost = cache_hit_tokens * read_rate

        # Bill cache writes (typically 125% of input cost for Anthropic)
        if cache_write_tokens:
            write_rate = cache_write_rate or (input_cost_per_token * 1.25)
            cache_write_cost = cache_write_tokens * write_rate

    else:
        # LITELLM / OPENAI / DEEPSEEK SEMANTICS
        # prompt_tokens includes cache hits (if any)

        if cache_read_rate and cache_hit_tokens:
            # Separate pricing for cache reads
            non_cached = max(0, prompt_tokens - cache_hit_tokens)
            transient_tokens = non_cached
            input_cost = non_cached * input_cost_per_token
            cache_read_cost = cache_hit_tokens * cache_read_rate
        else:
            # No cache read discount - bill everything at input rate
            transient_tokens = prompt_tokens
            input_cost = prompt_tokens * input_cost_per_token

        # Cache writes (if tracked separately)
        if cache_write_tokens and cache_write_rate:
            cache_write_cost = cache_write_tokens * cache_write_rate

    return input_cost, cache_read_cost, cache_write_cost, transient_tokens


def calculate_cost_breakdown(
    prompt_tokens: int,
    completion_tokens: int,
    cache_hit_tokens: int,
    cache_write_tokens: int,
    model_info: Dict[str, Any],
    model_name: str = "",
    usage_format: str = 'unknown',
) -> Dict[str, Any]:
    """Calculate a detailed cost breakdown for a completion.

    Returns a dictionary with:
    - input_cost: Cost for non-cached input tokens
    - output_cost: Cost for output tokens
    - cache_read_cost: Cost for cache read tokens
    - cache_write_cost: Cost for cache write tokens
    - total_cost: Sum of all costs
    - transient_tokens: Non-cache input tokens (Anthropic-style)
    """
    input_cost_per_token = model_info.get("input_cost_per_token") or 0
    output_cost_per_token = model_info.get("output_cost_per_token") or 0

    cache_read_rate, cache_write_rate = get_cache_rates(model_info)
    is_anthropic_provider, is_bedrock_anthropic = detect_provider_style(model_info, model_name)

    # Only native Anthropic uses Anthropic-style semantics
    is_anthropic_style = is_anthropic_provider

    provider = (model_info.get("litellm_provider") or "").lower()

    # Calculate input costs
    input_cost, cache_read_cost, cache_write_cost, transient_tokens = calculate_input_cost(
        prompt_tokens=prompt_tokens,
        cache_hit_tokens=cache_hit_tokens,
        cache_write_tokens=cache_write_tokens,
        input_cost_per_token=input_cost_per_token,
        cache_read_rate=cache_read_rate,
        cache_write_rate=cache_write_rate,
        is_anthropic_style=is_anthropic_style,
        provider=provider,
        model_name=model_name,
        usage_format=usage_format,
    )

    # Calculate output cost
    output_cost = completion_tokens * output_cost_per_token

    total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cache_read_cost": cache_read_cost,
        "cache_write_cost": cache_write_cost,
        "total_cost": total_cost,
        "transient_tokens": transient_tokens,
    }


def compute_total_cost(
    prompt_tokens: int,
    completion_tokens: int,
    cache_write_tokens: int,
    cache_hit_tokens: int,
    model_info: Dict[str, Any],
    model_name: str = "",
    usage_format: str = 'unknown',
) -> float:
    """Compute the total cost for a completion (backwards compatible).

    This is the main entry point for cost calculation, returning just the total.
    For detailed breakdown, use calculate_cost_breakdown().
    """
    breakdown = calculate_cost_breakdown(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_hit_tokens=cache_hit_tokens,
        cache_write_tokens=cache_write_tokens,
        model_info=model_info,
        model_name=model_name,
        usage_format=usage_format,
    )
    return breakdown["total_cost"]
