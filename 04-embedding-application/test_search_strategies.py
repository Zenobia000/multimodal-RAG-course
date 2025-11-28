#!/usr/bin/env python3
"""
Vector Search Strategies Test Module
Test various retrieval strategies functionality and performance
"""

import json
import pytest
import numpy as np
from typing import List, Dict, Optional, Tuple
from unittest.mock import Mock, patch
import warnings
warnings.filterwarnings('ignore')

# Import required libraries for testing
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from openai import OpenAI

# Mock configuration
TEST_CONFIG = {
    'QDRANT_URL': 'http://localhost:6333',
    'COLLECTION_NAME': 'test_collection',
    'EMBEDDING_MODEL': 'text-embedding-ada-002'
}

class TestVectorSearchStrategies:
    """Vector search strategies test class"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Test setup"""
        self.mock_client = Mock(spec=QdrantClient)
        self.mock_embeddings = Mock(spec=OpenAIEmbeddings)
        self.mock_openai_client = Mock(spec=OpenAI)

        # Mock vectors
        self.mock_vector = [0.1] * 1536
        self.mock_embeddings.embed_query.return_value = self.mock_vector

        # Mock search results
        self.mock_search_result = [
            Mock(
                id='test_id_1',
                score=0.95,
                payload={
                    'page_content': 'Test content about transformers and attention mechanisms',
                    'metadata': {
                        'source_file': 'test_file_1.pdf',
                        'chunk_size': 1000,
                        'chunk_id': 'chunk_1'
                    }
                }
            ),
            Mock(
                id='test_id_2',
                score=0.87,
                payload={
                    'page_content': 'Deep learning neural networks with scaling laws',
                    'metadata': {
                        'source_file': 'test_file_2.pdf',
                        'chunk_size': 950,
                        'chunk_id': 'chunk_2'
                    }
                }
            )
        ]

        self.mock_client.search.return_value = self.mock_search_result
        self.mock_client.get_collection.return_value = Mock(
            points_count=1000,
            config=Mock(
                params=Mock(
                    vectors=Mock(size=1536, distance='Cosine')
                )
            ),
            status='green'
        )

    def test_semantic_search(self):
        """Test semantic search functionality"""
        def semantic_search(query: str, limit: int = 5, score_threshold: float = 0.7):
            query_vector = self.mock_embeddings.embed_query(query)
            search_result = self.mock_client.search(
                collection_name=TEST_CONFIG['COLLECTION_NAME'],
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            return search_result

        # Execute test
        results = semantic_search("test query", limit=2)

        # Verify
        assert len(results) == 2
        assert results[0].score >= results[1].score  # Check sorting
        assert all(result.score >= 0.7 for result in results)  # Check threshold
        self.mock_embeddings.embed_query.assert_called_once_with("test query")
        self.mock_client.search.assert_called_once()

    def test_filtered_search(self):
        """Test metadata filtering search"""
        def filtered_search(
            query: str,
            source_filter: Optional[str] = None,
            chunk_size_range: Optional[Tuple[int, int]] = None,
            limit: int = 5
        ):
            query_vector = self.mock_embeddings.embed_query(query)

            filter_conditions = []
            if source_filter:
                filter_conditions.append(
                    rest.FieldCondition(
                        key="metadata.source_file",
                        match=rest.MatchText(text=source_filter)
                    )
                )

            if chunk_size_range:
                min_size, max_size = chunk_size_range
                filter_conditions.append(
                    rest.FieldCondition(
                        key="metadata.chunk_size",
                        range=rest.Range(gte=min_size, lte=max_size)
                    )
                )

            search_filter = rest.Filter(must=filter_conditions) if filter_conditions else None

            return self.mock_client.search(
                collection_name=TEST_CONFIG['COLLECTION_NAME'],
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )

        # Test search with source file filter
        results = filtered_search(
            "test query",
            source_filter="test_file_1.pdf",
            limit=2
        )

        # Verify
        assert len(results) == 2
        self.mock_client.search.assert_called()
        _, kwargs = self.mock_client.search.call_args
        assert kwargs['query_filter'] is not None

    def test_hybrid_search(self):
        """Test hybrid search"""
        def keyword_score(text: str, keywords: List[str]) -> float:
            text_lower = text.lower()
            matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            return matches / len(keywords) if keywords else 0

        def hybrid_search(
            query: str,
            keywords: List[str],
            semantic_weight: float = 0.7,
            keyword_weight: float = 0.3,
            limit: int = 5
        ):
            # Mock hybrid search logic
            semantic_results = self.mock_search_result[:limit*2]

            hybrid_results = []
            for result in semantic_results:
                content = result.payload.get('page_content', '')
                semantic_score = result.score
                keyword_score_val = keyword_score(content, keywords)

                hybrid_score = (
                    semantic_weight * semantic_score +
                    keyword_weight * keyword_score_val
                )

                hybrid_results.append({
                    'result': result,
                    'hybrid_score': hybrid_score,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score_val
                })

            hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            return hybrid_results[:limit]

        # Execute test
        keywords = ["transformer", "attention"]
        results = hybrid_search("test query", keywords, limit=2)

        # Verify
        assert len(results) == 2
        assert all('hybrid_score' in result for result in results)
        assert results[0]['hybrid_score'] >= results[1]['hybrid_score']

        # Verify keyword scoring function
        test_text = "transformer attention mechanism"
        score = keyword_score(test_text, keywords)
        assert score == 1.0  # All keywords match

    @patch('openai.OpenAI')
    def test_query_expansion(self, mock_openai_class):
        """Test query expansion"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "expanded_queries": ["neural networks", "deep learning", "artificial intelligence"]
        })

        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai_instance

        def query_expansion(original_query: str):
            openai_client = OpenAI()

            prompt = f"Generate synonyms for: {original_query}"

            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=300
                )

                result = json.loads(response.choices[0].message.content)
                return result.get("expanded_queries", [original_query])
            except Exception:
                return [original_query]

        # Execute test
        expanded = query_expansion("machine learning")

        # Verify
        assert len(expanded) == 3
        assert "neural networks" in expanded
        mock_openai_instance.chat.completions.create.assert_called_once()

    def test_discovery_search(self):
        """Test discovery search"""
        def discovery_search(positive_examples: List[str], negative_examples: List[str] = None, limit: int = 5):
            if negative_examples is None:
                negative_examples = []

            positive_vectors = [self.mock_embeddings.embed_query(example) for example in positive_examples]
            negative_vectors = [self.mock_embeddings.embed_query(example) for example in negative_examples]

            return self.mock_client.discover(
                collection_name=TEST_CONFIG['COLLECTION_NAME'],
                positive=positive_vectors,
                negative=negative_vectors,
                limit=limit,
                with_payload=True
            )

        # Mock discover method
        self.mock_client.discover.return_value = self.mock_search_result

        # Execute test
        positive = ["transformer architecture"]
        negative = ["computer vision"]
        results = discovery_search(positive, negative, limit=2)

        # Verify
        assert len(results) == 2
        self.mock_client.discover.assert_called_once()
        _, kwargs = self.mock_client.discover.call_args
        assert len(kwargs['positive']) == 1
        assert len(kwargs['negative']) == 1

    def test_recommendation_search(self):
        """Test recommendation search"""
        def get_recommendations(document_ids: List[str], limit: int = 5):
            return self.mock_client.recommend(
                collection_name=TEST_CONFIG['COLLECTION_NAME'],
                positive=document_ids,
                limit=limit,
                with_payload=True
            )

        # Mock recommend method
        self.mock_client.recommend.return_value = self.mock_search_result

        # Execute test
        doc_ids = ["test_id_1", "test_id_2"]
        results = get_recommendations(doc_ids, limit=2)

        # Verify
        assert len(results) == 2
        self.mock_client.recommend.assert_called_once()
        _, kwargs = self.mock_client.recommend.call_args
        assert kwargs['positive'] == doc_ids

    def test_grouped_search(self):
        """Test grouped search"""
        # Mock grouped search results
        mock_group = Mock()
        mock_group.id = "test_file_1.pdf"
        mock_group.hits = [self.mock_search_result[0]]

        mock_grouped_results = Mock()
        mock_grouped_results.groups = [mock_group]

        self.mock_client.search_groups.return_value = mock_grouped_results

        def grouped_search(query: str, group_by_field: str, group_size: int = 2, limit: int = 6):
            query_vector = self.mock_embeddings.embed_query(query)

            return self.mock_client.search_groups(
                collection_name=TEST_CONFIG['COLLECTION_NAME'],
                query_vector=query_vector,
                group_by=group_by_field,
                group_size=group_size,
                limit=limit,
                with_payload=True
            )

        # Execute test
        results = grouped_search("test query", "metadata.source_file", group_size=2, limit=6)

        # Verify
        assert results is not None
        assert len(results.groups) == 1
        assert results.groups[0].id == "test_file_1.pdf"
        self.mock_client.search_groups.assert_called_once()

    @patch('openai.OpenAI')
    def test_llm_reranking(self, mock_openai_class):
        """Test LLM re-ranking"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "ranked_docs": [2, 1],
            "reasoning": "Document 2 is more relevant to the query"
        })

        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai_instance

        def llm_reranking(query: str, search_results: List, top_k: int = 3):
            openai_client = OpenAI()

            # Prepare candidate documents
            candidates = []
            for i, result in enumerate(search_results[:10]):
                content = result.payload['page_content'][:500]
                candidates.append(f"Document {i+1}: {content}")

            candidates_text = "\n\n".join(candidates)
            prompt = f"Rerank documents for query: {query}\n\n{candidates_text}"

            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300
                )

                result = json.loads(response.choices[0].message.content)
                ranked_indices = [int(idx) - 1 for idx in result.get("ranked_docs", [])]
                reasoning = result.get("reasoning", "No ranking reason")

                reranked_results = []
                for i, idx in enumerate(ranked_indices):
                    if 0 <= idx < len(search_results):
                        reranked_results.append({
                            'result': search_results[idx],
                            'original_rank': idx + 1,
                            'new_rank': i + 1,
                            'original_score': search_results[idx].score
                        })

                return reranked_results, reasoning

            except Exception:
                return search_results[:top_k], "Reranking failed"

        # Execute test
        reranked_results, reasoning = llm_reranking("test query", self.mock_search_result, top_k=2)

        # Verify
        assert len(reranked_results) == 2
        assert reasoning == "Document 2 is more relevant to the query"
        assert reranked_results[0]['new_rank'] == 1
        mock_openai_instance.chat.completions.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_adaptive_retrieval(self, mock_openai_class):
        """Test adaptive retrieval"""
        # Mock query analysis result
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "complexity_level": "simple",
            "query_type": "factual",
            "keywords_count": 2,
            "requires_multi_docs": False,
            "recommended_strategy": "semantic_search"
        })

        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai_instance

        def analyze_query_complexity(query: str):
            openai_client = OpenAI()

            prompt = f"Analyze query complexity: {query}"

            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=200
                )

                return json.loads(response.choices[0].message.content)
            except Exception:
                return {
                    "complexity_level": "simple",
                    "query_type": "factual",
                    "keywords_count": 1,
                    "requires_multi_docs": False,
                    "recommended_strategy": "semantic_search"
                }

        def adaptive_retrieval(query: str):
            analysis = analyze_query_complexity(query)
            complexity = analysis.get("complexity_level", "simple")

            # Select strategy based on complexity
            if complexity == "simple":
                strategy_used = "Semantic Search"
                results = self.mock_search_result[:3]
            else:
                strategy_used = "HyDE + LLM Reranking"
                results = self.mock_search_result[:5]

            return {
                'query': query,
                'analysis': analysis,
                'strategy_used': strategy_used,
                'results': results
            }

        # Execute test
        result = adaptive_retrieval("What is attention mechanism?")

        # Verify
        assert result['analysis']['complexity_level'] == 'simple'
        assert result['strategy_used'] == 'Semantic Search'
        assert len(result['results']) == 2  # mock_search_result has 2 items
        mock_openai_instance.chat.completions.create.assert_called_once()

    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        import time

        def calculate_metrics(search_function, query: str, iterations: int = 3):
            """Calculate performance metrics for search function"""
            times = []
            scores = []

            for _ in range(iterations):
                start_time = time.time()
                results = search_function(query)
                end_time = time.time()

                execution_time = end_time - start_time
                times.append(execution_time)

                # Calculate average score
                if hasattr(results[0], 'score'):
                    avg_score = np.mean([r.score for r in results])
                else:
                    avg_score = np.mean([r.get('score', 0) for r in results])
                scores.append(avg_score)

            return {
                'avg_execution_time': np.mean(times),
                'avg_score': np.mean(scores),
                'score_std': np.std(scores),
                'time_std': np.std(times)
            }

        # Mock search function
        def mock_search_function():
            time.sleep(0.01)  # Simulate search time
            return self.mock_search_result

        # Execute test
        metrics = calculate_metrics(mock_search_function, "test query", iterations=2)

        # Verify
        assert 'avg_execution_time' in metrics
        assert 'avg_score' in metrics
        assert metrics['avg_execution_time'] > 0
        assert metrics['avg_score'] > 0

    def test_error_handling(self):
        """Test error handling"""
        # Mock search failure
        self.mock_client.search.side_effect = Exception("Connection failed")

        def robust_search(query: str, fallback_strategy: str = "simple"):
            try:
                query_vector = self.mock_embeddings.embed_query(query)
                results = self.mock_client.search(
                    collection_name=TEST_CONFIG['COLLECTION_NAME'],
                    query_vector=query_vector,
                    limit=5,
                    with_payload=True
                )
                return results
            except Exception as e:
                # Use fallback strategy
                return {
                    'error': str(e),
                    'fallback_used': fallback_strategy,
                    'results': []
                }

        # Execute test
        result = robust_search("test query")

        # Verify error handling
        assert 'error' in result
        assert result['fallback_used'] == 'simple'
        assert result['results'] == []


class TestSearchStrategyIntegration:
    """Search strategy integration test"""

    def test_strategy_combination(self):
        """Test strategy combination"""
        # Mock combined strategies results
        def combined_search(query: str, strategies: List[str]):
            results = {}

            if 'semantic' in strategies:
                results['semantic'] = [{'score': 0.95, 'content': 'semantic result'}]

            if 'hybrid' in strategies:
                results['hybrid'] = [{'score': 0.88, 'content': 'hybrid result'}]

            if 'expanded' in strategies:
                results['expanded'] = [{'score': 0.91, 'content': 'expanded result'}]

            return results

        # Execute test
        strategies = ['semantic', 'hybrid', 'expanded']
        results = combined_search("test query", strategies)

        # Verify
        assert len(results) == 3
        assert all(strategy in results for strategy in strategies)
        assert all(len(results[strategy]) > 0 for strategy in strategies)

    def test_result_fusion(self):
        """Test result fusion"""
        def fuse_results(result_sets: List[List[Dict]], weights: List[float]):
            """Fuse multiple search result sets"""
            if len(result_sets) != len(weights):
                raise ValueError("Number of result sets must equal number of weights")

            # Simplified fusion logic
            fused = []
            for i, (results, weight) in enumerate(zip(result_sets, weights)):
                for result in results:
                    fused_result = result.copy()
                    fused_result['weighted_score'] = result.get('score', 0) * weight
                    fused_result['source_strategy'] = f'strategy_{i}'
                    fused.append(fused_result)

            # Sort by weighted score
            fused.sort(key=lambda x: x['weighted_score'], reverse=True)
            return fused

        # Test data
        result_set_1 = [{'score': 0.9, 'id': 'doc1'}, {'score': 0.8, 'id': 'doc2'}]
        result_set_2 = [{'score': 0.85, 'id': 'doc3'}, {'score': 0.75, 'id': 'doc4'}]
        weights = [0.6, 0.4]

        # Execute test
        fused_results = fuse_results([result_set_1, result_set_2], weights)

        # Verify
        assert len(fused_results) == 4
        assert fused_results[0]['weighted_score'] >= fused_results[1]['weighted_score']
        assert all('source_strategy' in result for result in fused_results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])