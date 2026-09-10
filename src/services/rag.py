### Core modules ###
import os
from typing import Optional, List


### Type hints ###


### Internal modules ###
from .base import ServiceBase
from .vector_database import VectorDatabase


class RAGBase(ServiceBase):
    def __init__(
        self,
        vector_database: VectorDatabase,
        **kwargs
    ):
        """
        Context retriever service.

        All retrieval config is sourced from the environment (`.env`) 
        see `.env.example` for the keys.

        Required environment keys:
            RAG_TOPK            (int):  contexts kept when reranking is disabled.
            RAG_RERANK_ENABLED  (bool): two-stage retrieval (dense pool -> cross-encoder).
            RAG_CANDIDATE_POOL  (int):  dense candidates fetched before reranking.
            RAG_RERANK_TOPK     (int):  contexts kept after reranking.

        Args:
            vector_database (VectorDatabase): vector database.
        """
        super().__init__(**kwargs)

        self.vector_database = vector_database

        # Retrieval tuning — read strictly from .env (no fallback defaults).
        self.topk           = int(os.environ["RAG_TOPK"])
        self.rerank_enabled = os.environ["RAG_RERANK_ENABLED"].strip().lower() == "true"
        self.candidate_pool = int(os.environ["RAG_CANDIDATE_POOL"])
        self.rerank_topk    = int(os.environ["RAG_RERANK_TOPK"])


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
        document_ids: Optional[List[str]] = None,
    ):
        # Qdrant filter spanning active memory scopes as a union
        # When document_ids is given the session scope is narrowed to those
        # documents so a file question targets that file's chunks
        from qdrant_client import models

        def eq(field: str, value):
            return models.FieldCondition(
                key=f"metadata.{field}", match=models.MatchValue(value=value)
            )

        should = []

        if include_session and user_id and session_id:
            session_must = [
                eq("memory_type", "session"),
                eq("user_id", user_id),
                eq("session_id", session_id),
            ]
            if document_ids:
                session_must.append(
                    models.FieldCondition(
                        key="metadata.document_id",
                        match=models.MatchAny(any=list(document_ids)),
                    )
                )
            should.append(models.Filter(must=session_must))

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

    @staticmethod
    def _scope_rank(doc) -> int:
        """Priority rank for ordering: session (0) > user (1) > global (2)."""
        meta = getattr(doc, "metadata", None) or {}
        return {"session": 0, "user": 1, "global_memory": 2}.get(
            meta.get("memory_type"), 3
        )

    @staticmethod
    def _source_info(doc, score=None) -> dict:
        """Describe where a chunk came from, for citation in the final answer.

        Normalises the stored ``memory_type`` into a stable tier key plus a
        human-readable label and the source title (e.g. the attached file's
        name for a session chunk, or the service name for a global chunk).
        """
        meta = getattr(doc, "metadata", None) or {}
        memory_type = meta.get("memory_type")
        title = str(meta.get("title") or meta.get("file_name") or "")
        service_name = str(meta.get("service_name") or "")

        tier, label = {
            "session": ("session", "Attached file / session"),
            "user": ("user", "Your saved memory"),
            "global_memory": ("global", "Global knowledge base"),
        }.get(memory_type, ("unknown", "Unknown source"))

        return {
            "tier": tier,
            "label": label,
            "title": title,
            "service_name": service_name,
            "score": None if score is None else float(score),
        }

    def _select(self, pairs, keep_n):
        """Rank by relevance and take the top keep_n."""
        ordered = sorted(
            pairs,
            key=lambda p: (-p[1], self._scope_rank(p[0])),
        )
        return ordered[:keep_n]

    def _retrieve_pairs(
        self,
        user_prompt: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        global_service_names: Optional[List[str]] = None,
        include_session: bool = True,
        include_user: bool = False,
        include_global: bool = True,
        document_ids: Optional[List[str]] = None,
    ) -> list:
        """Fetch + rank the active memory scopes
        returning selected (doc, score) pairs."""
        memory_filter = self.build_union_filter(
            user_id=user_id,
            session_id=session_id,
            global_service_names=global_service_names,
            include_session=include_session,
            include_user=include_user,
            include_global=include_global,
            document_ids=document_ids,
        )

        # If no scope resolved, return empty rather than searching the whole collection
        # unfiltered (which would leak other users'/sessions'/global points)
        if memory_filter is None:
            return []

        # Fetch a larger candidate pool when reranking
        fetch_k = self.candidate_pool if self.rerank_enabled else self.topk
        retrieved_contents = self.vector_database.similarity_search_with_relevance_scores(
            query=user_prompt,
            k=fetch_k,
            filter=memory_filter,
        )

        # debug
        _scope = f"session={include_session} user={include_user} global={include_global}"
        def _dbg(doc, score):
            meta = getattr(doc, "metadata", None) or {}
            return (meta.get("memory_type"), round(float(score), 4),
                    doc.page_content[:45].replace("\n", " "))
        print(f"[rag] DENSE  ({_scope}) " + str(
            [_dbg(d, s) for d, s in sorted(retrieved_contents, key=lambda p: p[1], reverse=True)[:12]]
        ))

        if self.rerank_enabled and retrieved_contents:
            candidate_docs = [doc for doc, _ in retrieved_contents]
            rerank_scores = self.vector_database.rerank(
                user_prompt, [d.page_content for d in candidate_docs]
            )
            if rerank_scores:
                print(f"[rag] RERANK ({_scope}) " + str(sorted(
                    [_dbg(d, x) for d, x in zip(candidate_docs, rerank_scores)],
                    key=lambda t: t[1], reverse=True)[:12]
                ))
                selected = self._select(
                    list(zip(candidate_docs, rerank_scores)), self.rerank_topk
                )
                print(f"[rag] SELECT ({_scope}) " + str([_dbg(d, s) for d, s in selected]))
                return selected
            # Reranker unavailable — fall back to dense cosine order.
            return self._select(list(retrieved_contents), self.topk)

        return self._select(list(retrieved_contents), self.topk)

    @staticmethod
    def _doc_key(doc) -> tuple:
        """Identity of a chunk for de-duplication across two retrieval passes."""
        meta = getattr(doc, "metadata", None) or {}
        return (meta.get("document_id"), doc.page_content)

    def _format_selection(self, selected: list):
        """Turn selected (doc, score) pairs into (context, scores, sources)."""
        retrieved_docs = [doc for doc, _ in selected]
        retrieved_context_score = [score for _, score in selected]
        sources = [self._source_info(doc, score) for doc, score in selected]

        if not retrieved_docs:
            return "", [], []

        def _title(doc) -> str:
            meta = getattr(doc, "metadata", None) or {}
            return str(meta.get("title") or "")

        context = "".join([
            f"{_title(doc)}: " + doc.page_content.replace("\n", " ") + ". "
            for doc in retrieved_docs
        ])
        return context, retrieved_context_score, sources

    def __call__(
        self,
        user_prompt: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        global_service_names: Optional[List[str]] = None,
        include_session: bool = True,
        include_user: bool = False,
        include_global: bool = True,
        document_ids: Optional[List[str]] = None,
    ):
        """
        Retrieve context for a given user prompt (single union query over the
        active memory tiers).

        Args:
            user_prompt (str): a question from the user.
            document_ids (list[str], optional): when a file is attached, narrow the
                session scope to these document ids so the file's chunks are the
                retrieval target.

        Returns:
            context             (str): retrieved context from the vector database.
            retrieved_doc_score      : score of each retrieved context chunk.
            sources             (list): per-chunk origin descriptors (tier, label,
                title, service_name, score) in the same order as the context, for
                citing where each piece of information came from.
        """
        selected = self._retrieve_pairs(
            user_prompt,
            user_id=user_id,
            session_id=session_id,
            global_service_names=global_service_names,
            include_session=include_session,
            include_user=include_user,
            include_global=include_global,
            document_ids=document_ids,
        )
        return self._format_selection(selected)

    def retrieve_with_attachment(
        self,
        user_prompt: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        global_service_names: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        include_user: bool = False,
        include_global: bool = True,
        file_directed: bool = False,
        force_kb: bool = False,
        kb_service_name: Optional[str] = None,
    ):
        """
        Retrieval for a turn with an attached file.

        The attached file's chunks are retrieved separately (so a generic
        question like "summarise this" always has them available), and the rest
        of the active memory (user + global) is retrieved too. What ends up in
        the context is decided purely by INTENT signals:

        - ``file_directed`` — the query points at the attached file.
        - ``force_kb`` — the query analyser routed the query to a memory-capable
          service (that service's knowledge base is what it wants).

        Rules:
        - The file is included unless the question does not reference it and
          was routed to a KB service
        - When the file is included AND the query was routed to a KB service,
          both are merged (file first). Otherwise the answer stays file-only.

        Returns (context, scores, sources)
        """
        # 1) Attached document's best chunks (file/session scope).
        doc_pairs = self._retrieve_pairs(
            user_prompt,
            user_id=user_id,
            session_id=session_id,
            global_service_names=global_service_names,
            include_session=True,
            include_user=False,
            include_global=False,
            document_ids=document_ids,
        )

        # 2) The rest 
        kb_scope = (
            [kb_service_name]
            if (force_kb and kb_service_name)
            else global_service_names
        )
        other_pairs = self._retrieve_pairs(
            user_prompt,
            user_id=user_id,
            session_id=session_id,
            global_service_names=kb_scope,
            include_session=False,
            include_user=include_user,
            include_global=include_global,
            document_ids=None,
        )

        # File not indexed / scoping mismatch — fall back to the full union so a
        # question is still answerable rather than silently returning nothing.
        if not doc_pairs:
            return self._format_selection(other_pairs)

        # The decision is driven purely by INTENT signals
        #   - file_directed : the query points at the attached file.
        #   - force_kb      : the query analyser routed this query to a
        #                     memory-capable service (its knowledge base).
        #
        # Include the file unless this is a pure domain question (routed to a KB
        # service) that does not reference the file — i.e. a stale attachment 
        # In that case answer from the knowledge base and do not cite the file.
        include_file = file_directed or not force_kb

        if not include_file:
            return self._format_selection(other_pairs)

        seen = {self._doc_key(d) for d, _ in doc_pairs}
        if force_kb:
            merged = list(doc_pairs) + [
                (d, s) for d, s in other_pairs if self._doc_key(d) not in seen
            ]
        else:
            merged = list(doc_pairs)
        return self._format_selection(merged)
