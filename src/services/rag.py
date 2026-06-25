### Core modules ###
from typing import Optional, List


### Type hints ###


### Internal modules ###
from .base import ServiceBase
from .vector_database import VectorDatabase


class RAGBase(ServiceBase):
    def __init__(
        self,
        vector_database: VectorDatabase,
        retrieve_score_threshold: float = 0.7,
        topk: int = 5,
        rerank_enabled: bool = True,
        candidate_pool: int = 30,
        rerank_topk: int = 6,
        rerank_score_threshold: float = 0.0,
        **kwargs
    ):
        """
        Context retriever service.

        Args:
            vector_database             (VectorDatabase):   vector database.
            retrieve_score_threshold    (float, optional):  retrieve score threshold to filter out retrieved context with
                                                            similarity under this threshold. Defaults to 0.7.
            topk                        (int, optional):    up to topk retrieved context returned. Defaults to 5.
            rerank_enabled              (bool, optional):   two-stage retrieval (dense pool -> cross-encoder). Default True.
            candidate_pool              (int, optional):    dense candidates fetched before reranking. Default 30.
            rerank_topk                 (int, optional):    contexts kept after reranking. Default 6.
            rerank_score_threshold      (float, optional):  drop reranked contexts under this score.
        """
        super().__init__(**kwargs)

        # Set config.
        self.retrieve_score_threshold = retrieve_score_threshold
        self.topk = topk
        self.vector_database = vector_database

        # Reranking config.
        self.rerank_enabled = rerank_enabled
        self.candidate_pool = candidate_pool
        self.rerank_topk = rerank_topk
        self.rerank_score_threshold = rerank_score_threshold


    def set_vector_database(
        self,
        vector_database: VectorDatabase
    ):
        """
        Set the vector database externally on demand.

        Args:
            vector_database (VectorDatabase): an external vector database.
        """
        self.vector_database = vector_database


    def set_retrieve_score_threshold(
        self,
        retrieve_score_threshold: float
    ):
        """
        Change the retrieve score externally on demand.

        Args:
            retrieve_score_threshold (float): an external threshold.
        """
        self.retrieve_score_threshold = retrieve_score_threshold


    def set_topk(
        self,
        topk: int
    ):
        """
        Change the topk externally on demand.

        Args:
            topk (int): topk documents to be retrieved.
        """
        self.topk = topk


    @staticmethod
    def build_memory_filter(
        memory_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        service_name: Optional[str] = None,
    ):
        # Build a Qdrant payload filter to scope retrieval by memory tier
        from qdrant_client import models

        must = []

        def eq(field: str, value):
            return models.FieldCondition(
                key=f"metadata.{field}",
                match=models.MatchValue(value=value),
            )

        if memory_type:
            must.append(eq("memory_type", memory_type))
        if user_id:
            must.append(eq("user_id", user_id))
        if session_id:
            must.append(eq("session_id", session_id))
        if service_name:
            must.append(eq("service_name", service_name))

        if not must:
            return None

        return models.Filter(must=must)

    @staticmethod
    def build_union_filter(
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        global_service_names: Optional[List[str]] = None,
        include_session: bool = True,
        include_user: bool = False,
        include_global: bool = True,
    ):
        # Qdrant filter spanning active memory scopes as a union
        from qdrant_client import models

        def eq(field: str, value):
            return models.FieldCondition(
                key=f"metadata.{field}", match=models.MatchValue(value=value)
            )

        should = []

        if include_session and user_id and session_id:
            should.append(models.Filter(must=[
                eq("memory_type", "session"),
                eq("user_id", user_id),
                eq("session_id", session_id),
            ]))

        if include_user and user_id:
            should.append(models.Filter(must=[
                eq("memory_type", "user"),
                eq("user_id", user_id),
            ]))

        if include_global and global_service_names:
            should.append(models.Filter(must=[
                eq("memory_type", "global_memory"),
                models.FieldCondition(
                    key="metadata.service_name",
                    match=models.MatchAny(any=list(global_service_names)),
                ),
            ]))

        if not should:
            return None

        return models.Filter(should=should)

    def __call__(
        self,
        user_prompt: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        global_service_names: Optional[List[str]] = None,
        include_session: bool = True,
        include_user: bool = False,
        include_global: bool = True,
    ):
        """
        Retrieve context for a given user prompt.

        Args:
            user_prompt (str): a question from the user.

        Returns:
            context             (str): retrieved context from the vector database.
            retrieved_doc_score      : score of the retrieved context.
        """
        # Scope retrieval to the union of active memory tiers
        memory_filter = self.build_union_filter(
            user_id=user_id,
            session_id=session_id,
            global_service_names=global_service_names,
            include_session=include_session,
            include_user=include_user,
            include_global=include_global,
        )

        # If no scope resolved, return empty rather than searching the whole collection 
        # unfiltered (which would leak other users'/sessions'/global points)
        if memory_filter is None:
            return "", []

        # Fetch a larger candidate pool when reranking
        fetch_k = self.candidate_pool if self.rerank_enabled else self.topk
        retrieved_contents = self.vector_database.similarity_search_with_relevance_scores(
            query=user_prompt,
            k=fetch_k,
            filter=memory_filter,
        )

        # Rerank with the cross-encoder, if enabled.
        retrieved_docs: list = []
        retrieved_context_score: list = []

        if self.rerank_enabled and retrieved_contents:
            candidate_docs = [doc for doc, _ in retrieved_contents]
            rerank_scores = self.vector_database.rerank(
                user_prompt, [d.page_content for d in candidate_docs]
            )

            if rerank_scores:
                ranked = sorted(
                    zip(candidate_docs, rerank_scores),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                for doc, score in ranked[: self.rerank_topk]:
                    if score >= self.rerank_score_threshold:
                        retrieved_docs.append(doc)
                        retrieved_context_score.append(score)
            else:
                # Reranker unavailable fall back to dense cosine
                for doc, score in retrieved_contents[: self.topk]:
                    if score >= self.retrieve_score_threshold:
                        retrieved_docs.append(doc)
                        retrieved_context_score.append(score)
        else:
            # keep contexts above the cosine threshold.
            for doc, score in retrieved_contents:
                if score >= self.retrieve_score_threshold:
                    retrieved_docs.append(doc)
                    retrieved_context_score.append(score)

        if not retrieved_docs:
            context = ""
        else:
            # Change the linechange to avoid messing up the print and log file.
            context = "".join([
                f"{doc.metadata.get('title') if getattr(doc, 'metadata', None) else None}: "
                + doc.page_content.replace("\n", " ")
                + ". "
                for doc in retrieved_docs
            ])

        return context, retrieved_context_score
