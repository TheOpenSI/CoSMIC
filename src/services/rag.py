### Core modules ###
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue
)


### Type hints ###
from typing import Any
from langchain_core.documents import Document


### Internal modules ###
from .base import ServiceBase
from .vector_database import VectorDatabase


class RAGBase(ServiceBase):
    def __init__(
        self,
        vector_database: VectorDatabase,
        topk: int = 5,
        rerank_enabled: bool = True,
        candidate_pool: int = 30,
        rerank_topk: int = 6,
        **kwargs: Any
    ) -> None:
        """
        Context retriever service.

        Args:
            vector_database (VectorDatabase): vector database.
            topk            (int, optional):  contexts kept when reranking is disabled. Defaults to 5.
            rerank_enabled  (bool, optional): two-stage retrieval (dense pool -> cross-encoder).
                                              Defaults to True.
            candidate_pool  (int, optional):  dense candidates fetched before reranking. Defaults to 30.
            rerank_topk     (int, optional):  contexts kept after reranking. Defaults to 6.
        """
        super().__init__(**kwargs)

        # Set config.
        self.topk               = topk
        self.vector_database    = vector_database

        # Reranking config.
        self.rerank_enabled         = rerank_enabled
        self.candidate_pool         = candidate_pool
        self.rerank_topk            = rerank_topk


    def set_vector_database(
        self,
        vector_database: VectorDatabase
    ) -> None:
        """
        Set the vector database externally on demand.

        Args:
            vector_database (VectorDatabase): an external vector database.
        """
        self.vector_database = vector_database


    @staticmethod
    def eq(
        field: str,
        value: str
    ) -> FieldCondition:
        return FieldCondition(
            key=f"metadata.{field}",
            match=MatchValue(value=value)
        )


    @staticmethod
    def _title(
        doc: Any
    ) -> str:
        meta: dict[str, Any] = getattr(doc, "metadata", {})

        return meta.get("title", "")


    @staticmethod
    def build_memory_filter(
        memory_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        service_name: str | None = None
    ) -> Filter | None:
        """
        Build a Qdrant payload filter scoping retrieval to a single memory tier.

        Args:
            memory_type (str, optional): memory tier (session, user, global_memory).
            user_id (str, optional): owner of the memory.
            session_id (str, optional): chat session of the memory.
            service_name (str, optional): service a global memory belongs to.

        Returns:
            memory_filter (Filter | None): filter, or None if no field is set.
        """
        must: list[FieldCondition | None] = []

        if memory_type:
            must.append(RAGBase.eq(field="memory_type", value=memory_type))

        if user_id:
            must.append(RAGBase.eq(field="user_id", value=user_id))

        if session_id:
            must.append(RAGBase.eq(field="session_id", value=session_id))

        if service_name:
            must.append(RAGBase.eq(field="service_name", value=service_name))

        if not must:
            return None

        memory_filter: Filter | None = Filter(must=must) # pyright: ignore[reportArgumentType]

        return memory_filter


    @staticmethod
    def build_union_filter(
        user_id: str | None = None,
        session_id: str | None = None,
        global_service_names: list[str] | None = None,
        include_session: bool = True,
        include_user: bool = False,
        include_global: bool = True,
        document_ids: list[str] | None = None
    ) -> Filter | None:
        """
        Build a Qdrant filter spanning the active memory tiers as a union.

        Args:
            user_id (str, optional): owner of the session and user memory.
            session_id (str, optional): chat session of the session memory.
            global_service_names (list[str], optional): services whose global
                memory is searched.
            include_session (bool, optional): search session memory. Defaults to True.
            include_user (bool, optional): search user memory. Defaults to False.
            include_global (bool, optional): search global memory. Defaults to True.
            document_ids (list[str], optional): narrow the session tier to these
                attached documents.

        Returns:
            memory_filter (Filter | None): filter, or None if no tier is active.
        """
        should: list[Filter | None] = []

        if include_session and user_id and session_id:
            session_must: list[FieldCondition | None] = [
                RAGBase.eq(field="memory_type", value="session"),
                RAGBase.eq(field="user_id", value=user_id),
                RAGBase.eq(field="session_id", value=session_id)
            ]
            if document_ids:
                session_must.append(
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchAny(any=list(document_ids))
                    )
                )
            should.append(Filter(must=session_must)) # pyright: ignore[reportArgumentType]

        if include_user and user_id:
            should.append(Filter(must=[
                RAGBase.eq(field="memory_type", value="user"),
                RAGBase.eq(field="user_id", value=user_id)
            ]))

        if include_global and global_service_names:
            should.append(Filter(must=[
                RAGBase.eq(field="memory_type", value="global_memory"),
                FieldCondition(
                    key="metadata.service_name",
                    match=MatchAny(any=list(global_service_names))
                )
            ]))

        if not should:
            return None

        memory_filter: Filter | None = Filter(should=should) # pyright: ignore[reportArgumentType]

        return memory_filter


    @staticmethod
    def _scope_rank(
        doc: Any
    ) -> int:
        """
        Priority rank of a chunk's memory tier.

        Args:
            doc (Document): retrieved chunk.

        Returns:
            rank (int): session (0) > user (1) > global (2) > unknown (3).
        """
        meta: dict[str, Any] = getattr(doc, "metadata", {})

        if "memory_type" in meta:
            match meta["memory_type"]:
                case "session":
                    return 0

                case "user":
                    return 1

                case "global_memory":
                    return 2

                case _:
                    return 3

        return 3


    @staticmethod
    def _select(
        pairs: list[tuple[Document, float]],
        keep_n: int
    ) -> list[tuple[Document, float]]:
        """
        Rank chunks by relevance score and keep the best ones.

        Ties on score are broken by memory tier (session > user > global), so a
        weakly related session chunk no longer beats a highly relevant global one.

        Args:
            pairs   (list[tuple[Document, float]]): (chunk, score) candidates.
            keep_n  (int):                          number of chunks to keep.

        Returns:
            selected (list[tuple[Document, float]]): top keep_n (chunk, score) pairs.
        """
        ordered: list[tuple[Document, float]] = sorted(
            pairs,
            key=lambda pair: (-pair[1], RAGBase._scope_rank(doc=pair[0]))
        )

        return ordered[:keep_n]


    def _retrieve_pairs(
        self,
        user_prompt: str,
        user_id: str | None = None,
        session_id: str | None = None,
        global_service_names: list[str] | None = None,
        include_session: bool = True,
        include_user: bool = False,
        include_global: bool = True,
        document_ids: list[str] | None = None
    ) -> list[tuple[Document, float]]:
        """
        Search the active memory tiers and select the most relevant chunks.

        Args:
            Same as `__call__()`.

        Returns:
            selected (list[tuple[Document, float]]): selected (chunk, score) pairs,
                most relevant first.
        """
        selected: list[tuple[Document, float]] = []

        # Scope retrieval to the union of active memory tiers.
        memory_filter: Filter | None = self.build_union_filter(
            user_id=user_id,
            session_id=session_id,
            global_service_names=global_service_names,
            include_session=include_session,
            include_user=include_user,
            include_global=include_global,
            document_ids=document_ids
        )

        # Never search the collection unfiltered: that would leak other
        # users', sessions' and services' points.
        if memory_filter is None:
            return selected

        # Fetch a larger candidate pool when reranking.
        fetch_k: int = self.candidate_pool if self.rerank_enabled else self.topk
        retrieved_contents: list[tuple[Document, float]] = (
            self.vector_database.similarity_search_with_relevance_scores(
                query=user_prompt,
                k=fetch_k,
                filter=memory_filter
            )
        )

        if self.rerank_enabled and retrieved_contents:
            candidate_docs: list[Document] = [doc for doc, _ in retrieved_contents]
            rerank_scores: list[float] = self.vector_database.rerank(
                user_prompt,
                [doc.page_content for doc in candidate_docs]
            )

            if rerank_scores:
                selected = RAGBase._select(
                    pairs=list(zip(candidate_docs, rerank_scores)),
                    keep_n=self.rerank_topk
                )

                return selected

        # Reranking disabled or unavailable: use the dense similarity order.
        selected = RAGBase._select(
            pairs=list(retrieved_contents),
            keep_n=self.topk
        )

        return selected


    @staticmethod
    def _format_context(
        selected: list[tuple[Document, float]]
    ) -> tuple[str, list[float]]:
        """
        Join selected chunks into one context string, each prefixed by its title.

        Args:
            selected (list[tuple[Document, float]]): selected (chunk, score) pairs.

        Returns:
            context                 (str):          retrieved context for the LLM.
            retrieved_context_score (list[float]):  score of each chunk.
        """
        context: str = "".join(
            f"{RAGBase._title(doc=doc)}: {doc.page_content.replace("\n", " ")}. "
            for doc, _ in selected
        )
        retrieved_context_score: list[float] = [score for _, score in selected]

        return (
            context,
            retrieved_context_score
        )


    def __call__(
        self,
        user_prompt: str,
        user_id: str | None = None,
        session_id: str | None = None,
        global_service_names: list[str] | None = None,
        include_session: bool = True,
        include_user: bool = False,
        include_global: bool = True,
        document_ids: list[str] | None = None
    ) -> tuple[str, list[float]]:
        """
        Retrieve context for a given user prompt.

        Args:
            user_prompt (str): a question from the user.
            user_id (str, optional): owner of the session and user memory.
            session_id (str, optional): chat session of the session memory.
            global_service_names (list[str], optional): services whose global
                memory is searched.
            include_session (bool, optional): search session memory. Defaults to True.
            include_user (bool, optional): search user memory. Defaults to False.
            include_global (bool, optional): search global memory. Defaults to True.
            document_ids (list[str], optional): narrow the session tier to these
                attached documents.

        Returns:
            context (str): retrieved context from the vector database.
            retrieved_context_score (list[float]): score of each retrieved chunk,
                most relevant first.
        """
        selected: list[tuple[Document, float]] = self._retrieve_pairs(
            user_prompt=user_prompt,
            user_id=user_id,
            session_id=session_id,
            global_service_names=global_service_names,
            include_session=include_session,
            include_user=include_user,
            include_global=include_global,
            document_ids=document_ids
        )

        return RAGBase._format_context(selected=selected)
