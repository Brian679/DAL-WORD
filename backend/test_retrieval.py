#!/usr/bin/env python
"""
Test script for real academic retrieval pipeline.
Verifies that papers are retrieved from multiple academic sources.
"""

import os
import sys
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from agent.research_layer import (
    retrieval_pipeline,
    verify_generated_citations,
    search_semantic_scholar,
    search_crossref,
    search_arxiv,
    search_pubmed,
    citation_string,
    build_citation_context,
)

def test_individual_sources():
    """Test each source independently."""
    print("\n" + "=" * 70)
    print("TESTING INDIVIDUAL ACADEMIC SOURCES")
    print("=" * 70)

    query = "machine learning healthcare diagnosis"

    print(f"\n1. Testing Semantic Scholar: '{query}'")
    try:
        papers = search_semantic_scholar(query, limit=3)
        print(f"   ✓ Retrieved {len(papers)} papers")
        for p in papers[:2]:
            print(f"     - {p.title[:60]}... ({p.year})")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    print(f"\n2. Testing Crossref: '{query}'")
    try:
        papers = search_crossref(query, rows=3)
        print(f"   ✓ Retrieved {len(papers)} papers")
        for p in papers[:2]:
            print(f"     - {p.title[:60]}... ({p.year})")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    print(f"\n3. Testing arXiv: '{query}'")
    try:
        papers = search_arxiv(query, limit=3)
        print(f"   ✓ Retrieved {len(papers)} papers")
        for p in papers[:2]:
            print(f"     - {p.title[:60]}... ({p.year})")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    print(f"\n4. Testing PubMed: '{query}'")
    try:
        papers = search_pubmed(query, limit=3)
        print(f"   ✓ Retrieved {len(papers)} papers")
        for p in papers[:2]:
            print(f"     - {p.title[:60]}... ({p.year})")
    except Exception as e:
        print(f"   ✗ Failed: {e}")


def test_retrieval_pipeline():
    """Test the complete retrieval pipeline."""
    print("\n" + "=" * 70)
    print("TESTING COMPLETE RETRIEVAL PIPELINE")
    print("=" * 70)

    topic = "machine learning in healthcare"
    print(f"\nRunning retrieval pipeline for: '{topic}'")
    print("This will search multiple academic sources and rank results...")

    try:
        result = retrieval_pipeline(topic=topic, document_id=999)
        
        print(f"\n✓ Pipeline completed successfully!")
        print(f"  - Topic: {result.topic}")
        print(f"  - Expanded queries: {len(result.expanded_queries)}")
        print(f"  - Total candidates: {len(result.papers)}")
        print(f"  - Top ranked papers: {len(result.top_papers)}")
        print(f"  - Embedding storage: {result.embedding_path}")

        print(f"\n📚 TOP 10 PAPERS (by relevance ranking):")
        print("-" * 70)
        for i, paper in enumerate(result.top_papers[:10], 1):
            print(f"\n{i}. {paper.title}")
            if paper.authors:
                authors = ", ".join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors += ", et al."
                print(f"   Authors: {authors}")
            print(f"   Year: {paper.year}, Journal: {paper.journal}, Source: {paper.source}")
            if paper.doi:
                print(f"   DOI: {paper.doi}")
            if paper.url:
                print(f"   URL: {paper.url[:70]}")
            if paper.abstract:
                abstract = paper.abstract[:120].strip() + "..." if len(paper.abstract) > 120 else paper.abstract
                print(f"   Abstract: {abstract}")

        print(f"\n" + "=" * 70)
        print("CITATION CONTEXT (formatted for grounding LLM):")
        print("=" * 70)
        context = build_citation_context(result.top_papers, max_items=10)
        print(context)

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


def test_citation_verification():
    """Test citation verification against Crossref."""
    print("\n" + "=" * 70)
    print("TESTING CITATION VERIFICATION")
    print("=" * 70)

    # Example text with citations (some real, some hallucinated)
    sample_text = """
    Recent studies show that machine learning improves diagnosis accuracy (Smith 2023, DOI: 10.1234/fake.001).
    The work by "Neural Networks for Healthcare Systems" (Johnson 2022) demonstrates significant improvements.
    Another paper, "Deep Learning in Medical Imaging" (2021), found similar results.
    The famous work of "Quantum Computing and Biology" by Chen (2024) suggests new approaches.
    According to the research "10.1038/s41591-021-01234-x", clinical validation is essential.
    """

    print(f"\nVerifying citations in sample text:")
    print("-" * 70)
    print(sample_text.strip())
    print("-" * 70)

    try:
        verifications = verify_generated_citations(sample_text)
        
        print(f"\n✓ Verification completed!")
        print(f"  Total citations found: {len(verifications)}")

        for i, v in enumerate(verifications, 1):
            status_emoji = "✓" if v.confidence >= 70 else "⚠" if v.confidence >= 30 else "✗"
            print(f"\n{i}. {status_emoji} {v.status.upper()} (confidence: {v.confidence}%)")
            print(f"   Title: {v.title or 'N/A'}")
            print(f"   DOI: {v.doi or 'N/A'}")
            if v.notes:
                print(f"   Notes: {v.notes}")

    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "🔬 " * 15)
    print("REAL ACADEMIC RETRIEVAL TEST SUITE")
    print("🔬 " * 15)

    # Run tests
    test_individual_sources()
    test_retrieval_pipeline()
    test_citation_verification()

    print("\n" + "=" * 70)
    print("✓ TEST SUITE COMPLETE")
    print("=" * 70)
