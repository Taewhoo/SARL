"""
Quick local test for generator template/tokenization logic that has been
breaking in production.  No GPU / retriever / vLLM needed.
"""
import sys, os

# ── 1. Template substitution ───────────────────────────────────────────────
FEEDBACK_FORMAT_TEMPLATE_FOR_GENERATOR = (
    "\n\nFeedback:\n{feedback}\n\n"
    "Now, revise your research process based on the feedback above. "
    "Use the local_knowledge_base_retrieval (and optionally get_document) tools to "
    "gather more information. "
    "When you have gathered enough information, provide your final answer in the format:\n"
    "Explanation: {your explanation}\n"
    "Exact Answer: {your succinct final answer}"
)

full_feedback = (
    "1. Essential Constraint(s):\n   - The question asks for the capital of France.\n"
    "2. Evaluation:\n   - The answer 'London' is incorrect.\n"
    "3. Judgment:\n   - INCOMPLETE"
)

print("=== Test 1: .format() with braces in template (should fail) ===")
try:
    result = FEEDBACK_FORMAT_TEMPLATE_FOR_GENERATOR.format(feedback=full_feedback)
    print("UNEXPECTED PASS — .format() did not raise")
except KeyError as e:
    print(f"KeyError as expected: {e}")

print("\n=== Test 2: .replace() with braces in template (should work) ===")
try:
    result = FEEDBACK_FORMAT_TEMPLATE_FOR_GENERATOR.replace("{feedback}", full_feedback)
    assert "{your explanation}" in result, "literal braces should survive"
    assert full_feedback in result, "feedback text should be inserted"
    print(f"OK — result length: {len(result)}")
    print(f"Preview: {result[:200]}...")
except Exception as e:
    print(f"FAIL: {e}")

# ── 2. Tokenizer return_dict=False ─────────────────────────────────────────
print("\n=== Test 3: apply_chat_template return_dict=False ===")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/mnt/weka/post_training/checkpoints/Solar-Open-100B",
        trust_remote_code=True,
    )

    tools = [{"type": "function", "function": {"name": "search", "description": "search",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}]
    chat = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello"}]

    # With return_dict=False (our fix)
    ids = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True,
                                  return_dict=False, tools=tools)
    assert isinstance(ids, list), f"Expected list, got {type(ids)}"
    assert all(isinstance(x, int) for x in ids), "All elements should be int"
    print(f"OK — {len(ids)} token ids, type: {type(ids)}")

    # Verify += works (the operation that was failing)
    prefix = tok.encode("1. Essential Constraint(s):\n", add_special_tokens=False)
    ids += prefix
    print(f"OK — after += prefix: {len(ids)} tokens")

    # With return_dict=True (the default that was breaking)
    result_dict = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, tools=tools)
    print(f"return_dict=True type: {type(result_dict)}, keys: {list(result_dict.keys()) if hasattr(result_dict, 'keys') else 'N/A'}")
    try:
        result_dict += prefix
        print("UNEXPECTED: += on BatchEncoding did not fail")
    except TypeError as e:
        print(f"Confirmed: += on BatchEncoding fails with: {e}")

    # Test base_conversation pattern (slicing)
    base_conv = [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": "I am a user."}]
    base_ids = tok.apply_chat_template(base_conv, add_generation_prompt=False,
                                       tokenize=True, return_dict=False, tools=tools)
    assert isinstance(base_ids, list), f"base_ids should be list, got {type(base_ids)}"

    extended = [*base_conv, {"role": "user", "content": "Search result here."}]
    ext_ids = tok.apply_chat_template(extended, add_generation_prompt=True,
                                      tokenize=True, return_dict=False, tools=tools)
    obs_ids = ext_ids[len(base_ids):]
    assert isinstance(obs_ids, list), f"obs_ids should be list, got {type(obs_ids)}"
    print(f"OK — slicing works: base={len(base_ids)}, extended={len(ext_ids)}, obs={len(obs_ids)}")

    # Test return_dict=True with return_assistant_tokens_mask (lines 710/738)
    multi = [*chat, {"role": "assistant", "content": "I'll help."}, {"role": "user", "content": "Thanks"}]
    enc = tok.apply_chat_template(multi, add_generation_prompt=False,
                                  return_dict=True, return_assistant_tokens_mask=True,
                                  tokenize=True, tools=tools)
    loss_mask = enc["assistant_masks"]
    response_ids = enc["input_ids"]
    print(f"return_dict=True works for response encoding: input_ids type={type(response_ids)}, len={len(response_ids)}, mask len={len(loss_mask)}")

    # Test batch call (line 973)
    batch = [chat, chat]  # list of conversations
    batch_ids = tok.apply_chat_template(batch, add_generation_prompt=True,
                                        tokenize=True, return_dict=False)
    assert isinstance(batch_ids, list), f"batch should be list, got {type(batch_ids)}"
    assert isinstance(batch_ids[0], list), f"batch[0] should be list, got {type(batch_ids[0])}"
    print(f"OK — batch call: {len(batch_ids)} conversations, first has {len(batch_ids[0])} tokens")

    print("\n=== ALL TESTS PASSED ===")

except ImportError as e:
    print(f"SKIP (transformers not available): {e}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nFAIL: {e}")
