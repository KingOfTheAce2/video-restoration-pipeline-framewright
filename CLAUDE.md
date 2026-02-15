# FrameWright

## Rules
- Batch parallel operations in one message
- No working files in root (use `/tests`, `/docs`, `/scripts`)
- Edit > create; do exactly what's asked, nothing more

## Qwen 7B Delegation (vLLM)
Delegate isolated tasks (boilerplate, refactors, tests) to local Qwen 2.5 Coder 7B:
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "prompt": "Task: <describe>\n\nContext:\n```python\n<code>\n```\n\nOutput:",
    "max_tokens": 4096,
    "temperature": 0.7,
    "stop": ["```\n\n", "\n\n#"]
  }' | jq -r '.choices[0].text'
```
**Max context**: 16k tokens for 7B. Review output before applying.
