import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileDeletedEvent
from watchdog.observers import Observer

from .document_index import DocumentIndex, DocumentMetadata

logger = logging.getLogger(__name__)


class StorageEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        document_index: DocumentIndex,
        watched_path: str,
        get_rag_required_services=None,
        get_service_id_by_name=None,
    ):
        self.document_index = document_index
        self.watched_path = watched_path
        self.get_rag_required_services = get_rag_required_services or (lambda: [5])
        self.get_service_id_by_name = get_service_id_by_name or (lambda x: None)
        self.event_log_path = os.path.join(
            os.path.dirname(watched_path), "storage_events.log"
        )

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return

        try:
            self.process_file(event.src_path)
        except Exception as e:
            logger.error(f"Error processing file creation event: {e}", exc_info=True)

    def process_file(self, file_path: str) -> None:
        """Process a single file for indexing.

        This is the core indexing logic, shared by both the real-time watchdog
        ``on_created`` handler and the startup ``sync_with_filesystem`` scan.

        Deduplication rules
        -------------------
        * If the file already carries a valid ``file_id`` prefix **and** that
          ``file_id`` is already present in the index → skip (already indexed).
        * If a document with the **same original filename** already exists under
          the **same service** in the index → skip (duplicate within service).
          Different services *are* allowed to hold files with the same name.
        * Otherwise, generate a new ``file_id``, rename the file on disk, and
          add it to the index.
        """
        rel_path = os.path.relpath(file_path, self.watched_path)
        parts = rel_path.split(os.sep)

        if len(parts) < 2:
            logger.warning(f"File not in service folder: {file_path}")
            return

        service_name = parts[0]
        filename = parts[-1]

        # Check if file already has file_id format
        file_id, original_filename = self._extract_file_id(filename)

        # If it already has a file_id, check if it's already indexed
        if file_id and file_id in self.document_index.documents:
            # Already in the index, no action needed
            return

        # Determine the original filename for dedup check
        # (if file_id was extracted, original_filename is already set;
        #  if not, the whole filename IS the original filename)
        dedup_filename = original_filename if file_id else filename

        # ---- Per-service filename deduplication ----
        # Same filename under the same service → reject and remove from disk.
        # Different services are allowed to have the same filename.
        if self.document_index.has_file_in_service(dedup_filename, service_name):
            logger.warning(
                f"Rejected duplicate: file '{dedup_filename}' already exists "
                f"under service '{service_name}'. Removing the duplicate from disk."
            )
            try:
                os.remove(file_path)
                logger.info(f"Deleted duplicate file from disk: {file_path}")
            except OSError as e:
                logger.error(f"Failed to delete duplicate file {file_path}: {e}")
            return

        renamed = False

        if not file_id:
            file_id = uuid.uuid4().hex[:8]
            original_filename = filename
            renamed = True

            # Rename file to add file_id prefix
            new_filename = f"{file_id}_{original_filename}"
            new_path = os.path.join(os.path.dirname(file_path), new_filename)

            try:
                os.rename(file_path, new_path)
                file_path = new_path
                logger.info(f"Renamed file: {filename} -> {new_filename}")
            except OSError as e:
                logger.error(f"Failed to rename file {file_path}: {e}")
                return

        # Check if service requires RAG
        rag_services = self.get_rag_required_services()
        service_id = self.get_service_id_by_name(service_name)

        # Fallback for academic_governance when service registry is not yet available
        if service_id is None and service_name.lower() == "academic_governance":
            service_id = 5

        if service_id is None:
            logger.warning(
                f"Service '{service_name}' not found in registry. Skipping indexing."
            )
            self._log_event(
                "created",
                file_path,
                file_id,
                original_filename,
                service_name,
                None,
                os.path.getsize(file_path),
                "skipped",
                renamed,
            )
            return

        if service_id not in rag_services:
            logger.warning(
                f"Service '{service_name}' ({service_id}) does not have RAG_Req enabled"
            )
            self._log_event(
                "created",
                file_path,
                file_id,
                original_filename,
                service_name,
                service_id,
                os.path.getsize(file_path),
                "skipped",
                renamed,
            )
            return

        # Create metadata and add to index
        self.document_index.add_document(
            document_id=file_id,
            file_name=original_filename,
            memory_type="global_memory",
            chunk_count=0,
            service_id=service_id,
            service_name=service_name,
        )

        self._log_event(
            "created",
            file_path,
            file_id,
            original_filename,
            service_name,
            service_id,
            os.path.getsize(file_path),
            "indexed",
            renamed,
        )

        logger.info(
            f"Indexed file: {original_filename} (id: {file_id}, service: {service_name})"
        )

    def on_deleted(self, event: FileDeletedEvent) -> None:
        if event.is_directory:
            return

        try:
            file_path = event.src_path
            filename = os.path.basename(file_path)

            file_id, original_filename = self._extract_file_id(filename)

            if not file_id:
                logger.warning(f"File does not have valid file_id format: {filename}")
                return

            # Remove from index
            if self.document_index.remove_document(file_id):
                self._log_event(
                    "deleted",
                    file_path,
                    file_id,
                    original_filename,
                    None,
                    None,
                    0,
                    "removed",
                    False,
                )
                logger.info(f"Removed document from index: {file_id}")
            else:
                logger.warning(f"Document not found in index: {file_id}")

        except Exception as e:
            logger.error(f"Error processing file deletion event: {e}", exc_info=True)

    def _extract_file_id(self, filename: str) -> tuple[Optional[str], Optional[str]]:
        """Extract file_id and original_filename from filename.

        Expected format: {file_id}_{original_filename}
        file_id is 8-character hex string

        Returns:
            Tuple of (file_id, original_filename) or (None, None) if format invalid
        """
        if "_" not in filename:
            return None, None

        parts = filename.split("_", 1)
        potential_id = parts[0]

        # Check if it's 8-char hex string
        if len(potential_id) == 8 and all(c in "0123456789abcdef" for c in potential_id):
            return potential_id, parts[1] if len(parts) > 1 else filename

        return None, None

    def _log_event(
        self,
        event_type: str,
        file_path: str,
        file_id: str,
        original_filename: str,
        service_name: Optional[str],
        service_id: Optional[int],
        file_size: int,
        status: str,
        renamed: bool,
    ) -> None:
        """Log storage event to file."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "path": file_path.replace(os.sep, "/"),
            "document_id": file_id,
            "original_filename": original_filename,
            "service_name": service_name,
            "service_id": service_id,
            "file_size": file_size,
            "status": status,
            "renamed": renamed,
        }

        try:
            os.makedirs(os.path.dirname(self.event_log_path), exist_ok=True)
            with open(self.event_log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


class StorageEventWatcher:
    def __init__(
        self,
        watched_path: str,
        document_index: DocumentIndex,
        get_rag_required_services=None,
        get_service_id_by_name=None,
    ):
        self.watched_path = watched_path
        self.document_index = document_index
        self.observer = Observer()
        self.event_handler = StorageEventHandler(
            document_index,
            watched_path,
            get_rag_required_services,
            get_service_id_by_name,
        )

    def start(self) -> None:
        """Start monitoring for file system changes."""
        if not os.path.exists(self.watched_path):
            os.makedirs(self.watched_path, exist_ok=True)
            try:
                os.chmod(self.watched_path, 0o777)
            except Exception:
                pass

        # Query services and pre-create folders for those with memory_capability=True
        try:
            from ..opensi_cosmic import _get_raw_services
            services = _get_raw_services()
            memory_services = [s for s in services if s.get("memory_capability") is True]

            # Fallback to academic_governance if no service with memory_capability=True is found
            if not memory_services:
                memory_services = [{"name": "academic_governance"}]

            for service in memory_services:
                service_name = service.get("name")
                if service_name:
                    folder_path = os.path.join(self.watched_path, service_name)
                    os.makedirs(folder_path, exist_ok=True)
                    try:
                        os.chmod(folder_path, 0o777)
                    except Exception:
                        pass
                    logger.info(f"Ensured global memory folder exists with write permission: {folder_path}")
        except Exception as e:
            logger.error(f"Failed to create service-specific global folders: {e}")

        self.observer.schedule(self.event_handler, self.watched_path, recursive=True)
        self.observer.start()
        logger.info(f"StorageEventWatcher started, monitoring: {self.watched_path}")

    def stop(self) -> None:
        """Stop monitoring."""
        self.observer.stop()
        self.observer.join()
        logger.info("StorageEventWatcher stopped")

    def sync_with_filesystem(self) -> None:
        """Sync document_index with actual filesystem state on startup.

        Two-phase sync:
        1. **Prune** – remove index entries whose files no longer exist on disk.
        2. **Discover** – walk the watched directory tree and call
           ``process_file`` for every file that is not yet in the index.
           ``process_file`` internally handles deduplication (by file_id and
           by filename-within-service), so it is safe to call unconditionally.
        """
        logger.info("Starting filesystem sync...")

        # Phase 1: prune stale entries
        self.document_index.sync_with_filesystem(self.watched_path)

        # Phase 2: discover and index files that exist on disk but not in the index
        if os.path.exists(self.watched_path):
            for root, _dirs, files in os.walk(self.watched_path):
                for file in files:
                    # Skip non-document files (e.g. event logs, index JSON)
                    if file == "storage_events.log":
                        continue
                    if file.endswith(".json"):
                        continue

                    file_path = os.path.join(root, file)
                    if not os.path.isfile(file_path):
                        continue

                    try:
                        self.event_handler.process_file(file_path)
                    except Exception as e:
                        logger.error(
                            f"Error syncing file {file_path} on startup: {e}",
                            exc_info=True,
                        )

        logger.info("Filesystem sync completed")
