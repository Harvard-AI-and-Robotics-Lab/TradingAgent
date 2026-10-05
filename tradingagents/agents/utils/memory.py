import chromadb
from chromadb.config import Settings
from openai import OpenAI
from datetime import datetime, timedelta
import os
import numpy as np
import math


class FinancialSituationMemory:
    def __init__(self, name, config, time_decay_factor=0.95):
        if config["backend_url"] == "http://localhost:11434/v1":
            # local / ollama embeddings: 768 dimensions, free
            # max-len=8192 in https://huggingface.co/nomic-ai/nomic-embed-text-v1
            self.embedding = "nomic-embed-text"
        else:
            # OpenAI embeddings: 1536 dimensions, ~$0.0004 per 1k tokens
            # max-len=8192 in https://platform.openai.com/docs/guides/embeddings
            self.embedding = "text-embedding-3-small"
        # Always use OpenAI for embeddings regardless of LLM provider/backend_url
        self.client = OpenAI(base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.get_or_create_collection(name=name)
        # Conservative character cap to stay under embedding model's 8k token limit.
        # Roughly 1 token ~ 3-4 chars; 20k chars ~ 5-7k tokens.
        self.embedding_max_chars = config.get("embedding_max_chars", 20000)
        self.time_decay_factor = time_decay_factor
        self.max_age_days = config.get("max_age_days", 90)

    def _truncate_for_embedding(self, text: str) -> str:
        max_chars = getattr(self, "embedding_max_chars", 20000)
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        # Keep both head and tail to preserve variety across sections
        print(f"    Text length for embedding: {len(text)} -> max {self.embedding_max_chars}")
        return text[:half] + "\n...\n" + text[-half:]

    def get_embedding(self, text):
        """Get OpenAI embedding for a text"""
        original_len = len(text)
        text = self._truncate_for_embedding(text)

        response = self.client.embeddings.create(
            model=self.embedding, input=text
        )  # len: 1536
        return response.data[0].embedding

    def add_situations(self, situations_and_advice, trade_date=None):
        """Add financial situations and their corresponding advice with date tracking.

        Parameters:
        - situations_and_advice: list of tuples (situation, recommendation)
        - trade_date: str or datetime, date of the trading decision (defaults to current date)
        """
        assert trade_date is not None, "trade_date is None"
        if trade_date is None:
            trade_date = datetime.now()
        elif isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d")

        situations = []
        advice = []
        ids = []
        improvements = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation, improvement) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            improvements.append(improvement)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{
                "recommendation": rec,
                "improvement": imp,
                "date": trade_date.strftime("%Y-%m-%d"),
                "timestamp": trade_date.timestamp()
            } for rec, imp in zip(advice, improvements)],
            embeddings=embeddings,
            ids=ids,
        )

    def calculate_time_weight(self, memory_timestamp, trade_date):
        """Calculate time-based weight for memory relevance.

        Parameters:
        - memory_timestamp: timestamp of when memory was created
        - trade_date: current trading date for comparison

        Returns:
        - weight: float between 0 and 1, where 1 is most recent
        """
        if isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d")

        memory_date = datetime.fromtimestamp(memory_timestamp)
        days_difference = (trade_date - memory_date).days

        # Exponential decay: weight decreases exponentially with time
        # For daily decay: weight = decay_factor^days_difference
        weight = self.time_decay_factor ** days_difference

        return max(weight, 0.01)  # Minimum weight to avoid zero relevance

    def get_memories(self, current_situation, trade_date, n_matches=5, min_similarity=0.1):
        """Find matching recommendations using OpenAI embeddings with time-weighted scoring.

        Parameters:
        - current_situation: string description of current market situation
        - trade_date: str or datetime, current trading date (defaults to today)
        - n_matches: number of matches to return after time weighting
        - min_similarity: minimum similarity threshold for inclusion

        Returns:
        - list of matched results sorted by time-weighted relevance score
        """

        if trade_date is None:
            ValueError("Warning: trade_date is None, using current date {}".format(trade_date))
        if isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d")

        self.remove_old_memories(trade_date)

        try:
            query_embedding = self.get_embedding(current_situation)
        except Exception as e:
            print(f"Error getting embedding for current_situation: {e}")
            return []

        # Get more results initially to allow for time weighting
        initial_matches = min(n_matches * 3, 50)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            similarity_score = 1 - results["distances"][0][i]

            # Skip memories below similarity threshold
            if similarity_score < min_similarity:
                continue

            metadata = results["metadatas"][0][i]

            # Calculate time weight
            memory_timestamp = metadata.get("timestamp", "unknown")
            if memory_timestamp == "unknown":
                time_weight = 1.0
            else:
                time_weight = self.calculate_time_weight(memory_timestamp, trade_date)

            # Combined relevance score: similarity * time_weight
            relevance_score = similarity_score * time_weight

            days_ago = (trade_date - datetime.fromtimestamp(memory_timestamp)).days if memory_timestamp != "unknown" else "unknown"

            matched_results.append({
                "matched_situation": results["documents"][0][i],
                "recommendation": metadata["recommendation"],
                "improvement": metadata["improvement"],
                "date": metadata.get("date", "unknown"),
                "similarity_score": similarity_score,
                "time_weight": time_weight,
                "relevance_score": relevance_score,
                "days_ago": days_ago
            })

        # Sort by relevance score (similarity * time weight) and return top n_matches
        matched_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return matched_results[:n_matches]

    def get_memories_by_date_range(self, current_situation, trade_date, start_date, end_date, n_matches=10):
        """Retrieve memories from a specific date range.

        Parameters:
        - current_situation: string description of current market situation
        - trade_date: str or datetime, current trading date (defaults to today)
        - start_date: str or datetime, start of date range
        - end_date: str or datetime, end of date range
        - n_matches: maximum number of memories to return

        Returns:
        - list of memories from the specified date range
        """
        if trade_date is None:
            ValueError("Warning: trade_date is None, using current date {}".format(trade_date))
        if isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d")

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        start_timestamp = start_date.timestamp()
        end_timestamp = end_date.timestamp()

        try:
            query_embedding = self.get_embedding(current_situation)
        except Exception as e:
            print(f"Error getting embedding for current_situation: {e}")
            return []

        # Get all memories to filter by date
        all_results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=self.situation_collection.count(),
            include=["metadatas", "documents"],
        )

        filtered_results = []
        for i, metadata in enumerate(all_results["metadatas"][0]):
            memory_timestamp = metadata.get("timestamp", "unknown")
            if memory_timestamp == "unknown":
                continue

            days_ago = (trade_date - datetime.fromtimestamp(memory_timestamp)).days

            if start_timestamp <= memory_timestamp <= end_timestamp:
                filtered_results.append({
                    "matched_situation": all_results["documents"][0][i],
                    "recommendation": metadata["recommendation"],
                    "improvement": metadata["improvement"],
                    "date": metadata.get("date", "unknown"),
                    "timestamp": memory_timestamp,
                    "days_ago": days_ago
                })

        # Sort by date (most recent first) and limit results
        filtered_results.sort(key=lambda x: x["timestamp"], reverse=True)
        return filtered_results[:n_matches]

    def get_memory_stats(self, current_situation, trade_date):
        """Get statistics about stored memories.

        Parameters:
        - current_situation: string description of current market situation
        - trade_date: str or datetime, current trading date (defaults to today)

        Returns:
        - dict with memory statistics including total count, date range, etc.
        """
        assert trade_date is not None, "trade_date is None"
        if isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d")

        total_count = self.situation_collection.count()
        if total_count == 0:
            return {"total_memories": 0, "date_range": None}

        try:
            query_embedding = self.get_embedding(current_situation)
        except Exception as e:
            print(f"Error getting embedding for current_situation: {e}")
            return {"total_memories": 0, "date_range": None}

        # Get all memories to analyze
        all_results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=total_count,
            include=["metadatas"],
        )

        timestamps = []
        for metadata in all_results["metadatas"][0]:
            timestamps.append(metadata.get("timestamp", trade_date.timestamp()))

        timestamps.sort()
        oldest_date = datetime.fromtimestamp(timestamps[0]).strftime("%Y-%m-%d")
        newest_date = datetime.fromtimestamp(timestamps[-1]).strftime("%Y-%m-%d")

        # Calculate average age
        current_timestamp = trade_date.timestamp()
        ages = [(current_timestamp - ts) / (24 * 3600) for ts in timestamps]  # days
        avg_age = sum(ages) / len(ages) if ages else 0

        return {
            "total_memories": total_count,
            "date_range": f"{oldest_date} to {newest_date}",
            "oldest_memory": oldest_date,
            "newest_memory": newest_date,
            "average_age_days": round(avg_age, 1),
            "time_decay_factor": self.time_decay_factor
        }

    def remove_old_memories(self, trade_date):
        """Remove memories older than max_age_days.

        Parameters:
        - trade_date: str or datetime, current trading date (defaults to today)
        - max_age_days: int, maximum age in days for memories to keep (default: 30)

        Returns:
        - dict with removed count and remaining count
        """
        if isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d")

        current_timestamp = trade_date.timestamp()
        cutoff_timestamp = current_timestamp - (self.max_age_days * 24 * 3600)

        # Get all memories
        total_count = self.situation_collection.count()
        if total_count == 0:
            return {"removed": 0, "remaining": 0}

        # Get all IDs and metadatas
        all_results = self.situation_collection.get(include=["metadatas"])

        ids_to_remove = []
        for i, metadata in enumerate(all_results["metadatas"]):
            timestamp = metadata.get("timestamp", current_timestamp)
            if timestamp < cutoff_timestamp:
                ids_to_remove.append(all_results["ids"][i])

        # Remove old memories
        if ids_to_remove:
            self.situation_collection.delete(ids=ids_to_remove)

        removed_count = len(ids_to_remove)
        remaining_count = total_count - removed_count

        print(f"Removed {removed_count} memories older than {self.max_age_days} days. {remaining_count} memories remaining.")

        # return {
        #     "removed": removed_count,
        #     "remaining": remaining_count,
        #     "cutoff_date": datetime.fromtimestamp(cutoff_timestamp).strftime("%Y-%m-%d")
        # }


if __name__ == "__main__":
    # Example usage with multi-day trading support
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from default_config import DEFAULT_CONFIG

    # Initialize memory with time decay factor (0.95 = 5% daily decay)
    matcher = FinancialSituationMemory("test_memory", DEFAULT_CONFIG, time_decay_factor=0.95)

    # Example data with different dates
    example_data_recent = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
    ]

    example_data_older = [
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
            "Reduce exposure to emerging market debt and increase exposure to developed market debt.",
        ),
    ]

    example_data_older = [
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
            "Increase exposure to sectors benefiting from higher rates and decrease exposure to sectors benefiting from lower rates.",
        ),
    ]

    # Add memories from different dates
    recent_date = "2024-01-15"
    older_date = "2024-01-01"

    matcher.add_situations(example_data_recent, trade_date=recent_date)
    matcher.add_situations(example_data_older, trade_date=older_date)

    # Example query with current date
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors
    reducing positions and rising interest rates affecting growth stock valuations
    """

    trade_date = "2024-01-16"

    try:
        # Get time-weighted recommendations
        recommendations = matcher.get_memories(
            current_situation,
            trade_date=trade_date,
            n_matches=3
        )

        print("=== Time-Weighted Memory Retrieval ===")
        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Date: {rec['date']} ({rec['days_ago']} days ago)")
            print(f"Similarity Score: {rec['similarity_score']:.3f}")
            print(f"Time Weight: {rec['time_weight']:.3f}")
            print(f"Relevance Score: {rec['relevance_score']:.3f}")
            print(f"Matched Situation: {rec['matched_situation'][:80]}...")
            print(f"Recommendation: {rec['recommendation'][:80]}...")

        # Get memory statistics
        stats = matcher.get_memory_stats(current_situation, trade_date)
        print(f"\n=== Memory Statistics ===")
        print(f"Total Memories: {stats['total_memories']}")
        print(f"Date Range: {stats['date_range']}")
        print(f"Average Age: {stats['average_age_days']} days")
        print(f"Time Decay Factor: {stats['time_decay_factor']}")

        # Get memories from date range
        print(f"\n=== Memories from Last 10 Days ===")
        start_date = "2024-01-06"
        end_date = "2024-01-16"
        range_memories = matcher.get_memories_by_date_range(start_date, end_date, n_matches=5)

        for i, mem in enumerate(range_memories, 1):
            print(f"{i}. Date: {mem['date']} - {mem['matched_situation'][:60]}...")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
