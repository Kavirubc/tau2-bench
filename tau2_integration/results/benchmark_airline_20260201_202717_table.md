# Benchmark Results Table

**File**: `benchmark_airline_20260201_202717.json`
**Model**: gemini-2.0-flash
**Domain**: airline
**Timestamp**: 2026-02-01T20:26:16.940626

| Task | Framework | Success | Tokens | Time (s) | LLM Calls | Tool Calls |
|------|-----------|---------|--------|----------|-----------|------------|
| 5 | LangGraph | ✓ | 3,788 | 1.57 | 1 | 0 |
| 5 | LG (PE) | ✓ | 4,000 | 1.66 | 1 | 0 |
| 5 | RAC | ✓ | 3,787 | 1.93 | 1 | 0 |
| 5 | SagaLLM | ✓ | 7,725 | 6.28 | 4 | 0 |
| 6 | LangGraph | ✓ | 3,712 | 1.35 | 1 | 0 |
| 6 | LG (PE) | ✓ | 8,090 | 2.80 | 2 | 1 |
| 6 | RAC | ✓ | 3,668 | 1.46 | 1 | 0 |
| 6 | SagaLLM | ✓ | 7,603 | 5.83 | 4 | 0 |
| 8 | LangGraph | ✓ | 3,883 | 1.57 | 1 | 0 |
| 8 | LG (PE) | ✓ | 4,109 | 1.68 | 1 | 0 |
| 8 | RAC | ✓ | 48,722 | 14.29 | 12 | 11 |
| 8 | SagaLLM | ✓ | 9,120 | 7.28 | 5 | 0 |
| 9 | LangGraph | ✓ | 3,859 | 2.21 | 1 | 0 |
| 9 | LG (PE) | ✓ | 12,810 | 3.78 | 3 | 2 |
| 9 | RAC | ✓ | 3,756 | 1.69 | 1 | 0 |
| 9 | SagaLLM | ✓ | 7,369 | 4.84 | 4 | 0 |

## Summary Statistics

| Framework | Success Rate | Avg Tokens | Avg Time (s) | Avg LLM Calls | Avg Tool Calls |
|-----------|--------------|------------|--------------|---------------|----------------|
| LangGraph | 4/4 (100.0%) | 3,810.5 | 1.68 | 1.0 | 0.0 |
| LG (PE) | 4/4 (100.0%) | 7,252.2 | 2.48 | 1.8 | 0.8 |
| RAC | 4/4 (100.0%) | 14,983.2 | 4.84 | 3.8 | 2.8 |
| SagaLLM | 4/4 (100.0%) | 7,954.2 | 6.06 | 4.2 | 0.0 |