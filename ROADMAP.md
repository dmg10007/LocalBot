# LocalBot Roadmap

## Goals

LocalBot is evolving into a local-first, model-agnostic assistant with specialized models, tool routing, GitHub integration, and an OpenWebUI front end. The main goals are faster responses, better tool use, lower external API cost, and stronger coding/research workflows.

---

## Current Direction

- Keep the architecture model-agnostic so models can be swapped and tested easily.
- Use specialized models for different tasks: general chat, coding, and reasoning.
- Prefer a flow where a tool-capable or research-capable model gathers information and passes distilled context to the coding model.
- Maintain a local-first design to reduce cloud dependence and control API costs.
- Preserve OpenWebUI as the main user-facing web interface.

---

## Priority Roadmap

### 1. Routing and Orchestration

- Add a lightweight router model in front of the main models.
- Classify each request into:
  - local-only response
  - coding task
  - reasoning task
  - Brave search
  - Perplexity research
  - GitHub/repo workflow
- Keep the router extremely fast and cheap so large models only run when needed.

### 2. Prompt Compression

- Add rolling conversation summaries to reduce prompt size each turn.
- Compress older history into a persistent session brief.
- Keep only the most recent turns plus the running summary in active context.
- Distill tool results (search, fetch, repo) before sending them to larger models.

### 3. Caching

- Cache tool results for repeated or near-duplicate searches.
- Cache Brave and Perplexity responses with configurable TTLs.
- Cache repo/workspace summaries per session.
- Reuse stable prompt prefixes to benefit from llama.cpp prompt caching.

### 4. Research Stack (Brave + Perplexity)

Use a cost-tiered escalation policy:

| Tier | Tool | Use When |
|------|------|----------|
| 1 | Local model | No fresh data needed |
| 2 | Brave Search | Fresh facts, current docs, pricing, release notes |
| 3 | Brave Grounding | Cited answer needed from live web |
| 4 | Perplexity Sonar | Deep multi-source synthesis, complex research |

**Brave** is the default live-web tool — cheap, fast, and independent index.
**Perplexity** is reserved for premium research only when synthesis quality justifies the cost.

Relevant `.env` variables to add:

```env
BRAVE_API_KEY=
PERPLEXITY_API_KEY=
SEARCH_RESULT_COUNT=5
SEARCH_FETCH_COUNT=3
SEARCH_FETCH_CHARS=1500
SEARCH_FETCH_TIMEOUT_SECONDS=8
RESEARCH_CACHE_TTL_SECONDS=600
```

### 5. Coding Workflow

- Build per-session repo digests so the coding model does not need full repo context every turn.
- Digest includes: repo structure, important files, recent changes, open issues/TODOs.
- Improve GitHub automation: branches, diffs, PRs, safe patch application.
- Maintain a local folder sandbox alongside GitHub-backed workflows.

### 6. OpenWebUI Experience

- Keep model capabilities separated by role:
  - **General** — assistant (chat, file reading, light research, memory)
  - **Coding** — builder (coding, GitHub, terminal, code interpreter)
  - **Reasoning** — analyst (deep analysis, architecture, tradeoffs, research synthesis)
- Expose "research mode" and "deep research mode" as explicit user-facing options.
- Add status updates for long-running steps: search, fetch, summarize, diff, patch.

### 7. Performance Improvements

- Favor stable system prompts and tool schemas for prefix reuse across turns.
- Reduce prompt bloat from repeated raw tool outputs.
- Tune context size, history count, and fetch sizes for lower prompt-eval time.
- Draft/final workflow option:
  - Fast model gives immediate provisional output.
  - Stronger model refines if needed.
- Batch concurrent tool calls where possible to reduce wall-clock latency.

### 8. Memory and Knowledge

- Separate short-term session memory from long-term stored memory.
- Store reusable user/project preferences:
  - preferred coding style and output format
  - repo-specific conventions
  - infrastructure constraints (e.g., 8–9 GB RAM limit)
- Use knowledge base notes for recurring project context instead of replaying it in prompts.

---

## Tool Strategy Summary

| Tool | Role | Cost |
|------|------|------|
| Local model | Default for all tasks | Free |
| Brave Search | Live web retrieval | Very low |
| Brave Grounding | Cited live-web answers | Low |
| Perplexity Sonar | Premium deep research | Medium (escalation only) |
| GitHub MCP | Repo read/write, PRs | Free (MCP) |
| Local filesystem | Sandbox coding | Free |

---

## Implementation Phases

### Phase 1 — Foundation
- [ ] Add router model or intent classifier
- [ ] Add rolling conversation summarizer
- [ ] Add Brave result cache with TTL
- [ ] Add repo digest generation per session

### Phase 2 — Research and Context
- [ ] Add Brave → Perplexity escalation policy
- [ ] Add explicit research modes in OpenWebUI
- [ ] Add session-level workspace summaries
- [ ] Add tool result distillation step

### Phase 3 — Performance and Memory
- [ ] Add semantic caching for repeated questions
- [ ] Add fast draft/final response workflow
- [ ] Add richer GitHub automation (diff review, patch, PR description)
- [ ] Add project-specific memory and coding preferences

---

## Success Metrics

- Lower time to first token
- Higher effective tokens per second
- Fewer unnecessary calls to large models
- Smaller average prompt size
- Better tool selection accuracy
- Lower paid API spend (Brave first, Perplexity only when needed)
- Better quality on research and coding tasks

---

## Guiding Principle

LocalBot should stay **local-first, tool-smart, and role-specialized**: local models for most work, Brave for cheap freshness, and Perplexity only when premium research quality is justified.
