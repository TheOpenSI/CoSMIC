### Core modules ###
import re
from dataclasses import (
    dataclass,
    field
)


### Type hints ###


### Internal modules ###
from .vector_database import VectorDatabase
from ...utils.log_tool import set_color


# Phrases that point the question at an attached file. A file merely named in
# the query (e.g. "the Academic Governance Handbook") is deliberately NOT a
# marker: that is a knowledge-base question. For the same reason, a bare
# "summarise" / "summary" / "the report" is not a marker ("summarise the
# Academic Governance Handbook" is a knowledge-base question); only forms that
# point at the user's own file are.
FILE_INTENT_MARKERS: tuple[str, ...] = (
    "this file", "the file", "this document", "the document",
    "this doc", "the doc", "this pdf", "the pdf",
    "this paper", "this report",
    "attached", "attachment", "uploaded",
    "summarise this", "summarize this", "summarise it", "summarize it",
    "summary of this", "summary of it",
)

# File refs are "<8-hex document_id>_<file name>".
FILE_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")

MISSING_FILE_LLM_NOTE: str = """\
IMPORTANT: the user's message refers to an attached file, but NO file is \
attached to this conversation. The Context below is from the shared \
knowledge base, which is NOT the user's attached file. Answer only the \
parts you can from that context, and for anything that requires the \
user's file, state that no file is attached and ask them to attach it.
"""

MISSING_FILE_USER_NOTE: str = """\
_You referred to an attached file, but no file is attached to this chat. \
I've answered from the knowledge base where I could — please attach the \
file so I can use it for the part of your question that needs it._\
"""


@dataclass
class AttachmentContext:
    """
    Files a question is about and how the question refers to them.

    Attributes:
        document_ids (list[str]): ids of the files retrieval should target.
        file_names (list[str]): display names, aligned with document_ids.
        file_directed (bool): the question explicitly points at a file.
        missing_file (bool): the question points at a file but none is available.
    """
    document_ids: list[str] = field(default_factory=list)
    file_names: list[str] = field(default_factory=list)
    file_directed: bool = False
    missing_file: bool = False


class AttachmentResolver:
    def __init__(
        self,
        vector_database: VectorDatabase
    ) -> None:
        """
        Resolve which attached files a question is about.

        Args:
            vector_database (VectorDatabase): used to look up files attached
                earlier in the session.
        """
        self.vector_database = vector_database


    @staticmethod
    def parse_file_refs(
        file_refs: list[str] | None
    ) -> tuple[list[str], list[str]]:
        """
        Split "<document_id>_<file name>" refs into ids and names.

        Malformed refs are skipped with a warning.

        Args:
            file_refs (list[str], optional): refs of the files attached this turn.

        Returns:
            document_ids (list[str]): parsed document ids.
            file_names (list[str]): file names, aligned with document_ids.
        """
        document_ids: list[str] = []
        file_names: list[str] = []

        for ref in (file_refs or []):
            document_id, _, file_name = ref.partition("_")
            if not FILE_ID_PATTERN.match(document_id):
                print(
                    set_color(
                        "warning",
                        f"[qa] WARNING: unparseable file ref '{ref}' (no 8-hex file_id); skipping."
                    )
                )
                continue
            document_ids.append(document_id)
            file_names.append(file_name or ref)

        return document_ids, file_names


    @staticmethod
    def is_file_directed(
        query: str
    ) -> bool:
        """
        Check whether a question explicitly points at an attached file.

        Args:
            query (str): a question.

        Returns:
            file_directed (bool): True if the query contains a file-intent marker.
        """
        query_lower = query.lower()
        return any(marker in query_lower for marker in FILE_INTENT_MARKERS)


    def get_session_files(
        self,
        user_id: str,
        session_id: str
    ) -> tuple[list[str], list[str]]:
        """
        Look up the files attached earlier in the session.

        Args:
            user_id (str): owner of the session.
            session_id (str): chat session.

        Returns:
            document_ids (list[str]): ids of the session's files.
            file_names (list[str]): file names, aligned with document_ids.
        """
        try:
            session_documents = self.vector_database.get_session_document_ids(
                user_id, session_id
            )
        except Exception as exc:
            print(
                set_color(
                    "warning",
                    f"[qa] WARNING: session document lookup failed: {exc}"
                )
            )
            return [], []

        document_ids = [document_id for document_id, _ in session_documents]
        file_names = [file_name for _, file_name in session_documents]
        return document_ids, file_names


    def __call__(
        self,
        query: str,
        file_refs: list[str] | None,
        has_files: bool,
        user_id: str | None,
        session_id: str | None
    ) -> AttachmentContext:
        """
        Resolve the files a question is about.

        Files attached this turn are used first, and the question is always
        treated as directed at them: refs only come from the current message,
        so a file attached this turn is never stale. A follow-up sent without
        an attachment ("summarise it") falls back to the files attached earlier
        in the session only when the query itself points at a file.

        Args:
            query (str): a question.
            file_refs (list[str], optional): refs of the files attached this turn.
            has_files (bool): files were attached this turn.
            user_id (str, optional): current user.
            session_id (str, optional): current chat session.

        Returns:
            attachment (AttachmentContext): resolved files and intent flags.
        """
        document_ids, file_names = self.parse_file_refs(file_refs)
        file_directed = bool(document_ids) or self.is_file_directed(query)

        if file_directed and not document_ids and user_id and session_id:
            document_ids, file_names = self.get_session_files(user_id, session_id)

        return AttachmentContext(
            document_ids=document_ids,
            file_names=file_names,
            file_directed=file_directed,
            missing_file=file_directed and not has_files and not document_ids,
        )
