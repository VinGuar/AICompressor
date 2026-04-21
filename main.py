#!/usr/bin/env python3
"""
ContextLite CLI — post-RAG context optimizer. No LLM calls.

Usage:
  python main.py --chunks "chunk1" "chunk2" --query "your question"
  python main.py --file chunks.txt --query "your question"
  python main.py --demo
"""

import argparse
import json
import sys


def print_result(result: dict, verbose: bool = False):
    ratio_pct = round(result["compression_ratio"] * 100, 1)
    saved = result["token_estimate_before"] - result["token_estimate_after"]

    print("\n" + "=" * 60)
    print("CONTEXTLITE RESULT")
    print("=" * 60)
    print(f"Tokens before : {result['token_estimate_before']:,}")
    print(f"Tokens after  : {result['token_estimate_after']:,}")
    print(f"Tokens saved  : {saved:,}  ({ratio_pct}% reduction)")
    print(f"Sentences kept: {len(result['kept_sentences'])}")
    print(f"Removed       : {len(result['removed_sentences'])}")
    print("-" * 60)
    print("OPTIMIZED CONTEXT:")
    print(result["optimized_context"])
    if result["explanation"]:
        print("-" * 60)
        print("WHAT HAPPENED:")
        for line in result["explanation"]:
            print(f"  • {line}")
    if verbose and result["removed_sentences"]:
        print("-" * 60)
        print("REMOVED SENTENCES:")
        for s in result["removed_sentences"]:
            print(f"  [-] {s[:120]}{'...' if len(s) > 120 else ''}")
    print("=" * 60 + "\n")


# Demo data: 5 chunks as a RAG system might return them for a SaaS pricing query.
# Chunks 1-2 are relevant, chunks 3-4 are mostly noise, chunk 5 duplicates chunk 1.
DEMO_CHUNKS = [
    # Chunk 0 — relevant
    "Our pricing starts at $499/month for the Starter plan. "
    "The Professional plan is $999/month. Enterprise is $2,499/month. "
    "Annual billing gives a 20% discount on all plans. "
    "A 14-day free trial is available with no credit card required.",

    # Chunk 1 — relevant (API limits)
    "The Starter plan includes 1,000 API requests per minute. "
    "Professional allows 10,000 requests per minute. "
    "Enterprise has no rate limits. "
    "All plans include REST and GraphQL API access.",

    # Chunk 2 — mostly noise for this query
    "Our company was founded in 2018 with offices in Austin, New York, and London. "
    "The CEO founded two previous companies before starting this one. "
    "We have raised $24M in Series B funding led by Sequoia Capital. "
    "Our engineering team is distributed across 12 time zones.",

    # Chunk 3 — near-duplicate of chunk 0 (tests deduplication)
    "Pricing for the platform begins at $499 per month on the Starter tier. "
    "Professional tier costs $999 monthly. The Enterprise tier is priced at $2,499/month. "
    "Customers on annual plans receive a 20% discount.",

    # Chunk 4 — mixed relevance
    "Customer support is available 24/7 via email and live chat. "
    "Support tickets are resolved within 4 hours on average. "
    "We have a 98% customer satisfaction score. "
    "The mobile app is available on iOS and Android and was released in 2021.",
]

DEMO_QUERY = "What are the pricing plans and API rate limits?"


def run_demo():
    from contextlite.pipeline import optimize
    print(f"\nQuery: '{DEMO_QUERY}'")
    print(f"Input: {len(DEMO_CHUNKS)} RAG-retrieved chunks\n")
    result = optimize(DEMO_CHUNKS, DEMO_QUERY, token_budget=300)
    print_result(result, verbose=True)


def main():
    parser = argparse.ArgumentParser(description="ContextLite — post-RAG context optimizer")
    parser.add_argument("--chunks", nargs="+", help="One or more chunk strings")
    parser.add_argument("--file", help="Path to file with one chunk per line (blank line = separator)")
    parser.add_argument("--query", help="The query this context will answer")
    parser.add_argument("--budget", type=int, default=2048, help="Token budget (default 2048)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Relevance threshold 0-1 (default 0.25)")
    parser.add_argument("--json-out", action="store_true", help="Output raw JSON")
    parser.add_argument("--verbose", action="store_true", help="Show removed sentences")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.chunks and not args.file:
        parser.print_help()
        sys.exit(1)

    if not args.query:
        print("Error: --query is required.")
        sys.exit(1)

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    else:
        chunks = args.chunks

    from contextlite.pipeline import optimize
    result = optimize(chunks, args.query, token_budget=args.budget, relevance_threshold=args.threshold)

    if args.json_out:
        print(json.dumps(result, indent=2))
    else:
        print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
