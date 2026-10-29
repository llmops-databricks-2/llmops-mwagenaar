"""HTTP client for the OpenBesluitvorming entity and snapshot endpoints."""

from __future__ import annotations

from http import HTTPStatus
from time import monotonic, sleep
from types import TracebackType
from typing import Self
from urllib.parse import quote

import httpx

from polderbot.ingestion.models import (
    AgendaItem,
    EntityDetail,
    MeetingDocument,
    SnapshotPage,
    SnapshotRecord,
)

BASE_URL = "https://openbesluitvorming.nl"
DEFAULT_SOURCE = "wetterskip_fryslan"
DEFAULT_EXPORT_LIMIT = 500
DEFAULT_SNAPSHOT_CURSOR = "meeting"
RATE_LIMIT_REQUEST_COST = 1.0

type AgendaItemPath = tuple[AgendaItem, ...]
type PendingAgendaItem = tuple[AgendaItem, AgendaItemPath]


class EntityNotFoundError(Exception):
    """Raised when an entity is not found in the OpenBesluitvorming API."""

    pass


class OpenBesluitvormingClient:
    """Retrieve OpenBesluitvorming entity details and snapshot exports.

    The client can be used as a context manager. It closes only an HTTP client
    that it created itself; caller-provided clients remain caller-managed.
    """

    def __init__(
        self,
        *,
        source: str = DEFAULT_SOURCE,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
        max_rate_limit_retries: int = 3,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            source: OpenBesluitvorming source key used for snapshot exports.
            base_url: Base URL used when creating an internal HTTP client.
            timeout: Timeout in seconds for requests made by an internal client.
            max_rate_limit_retries: Maximum number of retries after a
                rate-limit response.
            client: Optional caller-managed HTTP client. When omitted, this
                client creates and closes its own HTTP client.

        Raises:
            ValueError: If ``source`` is empty or ``max_rate_limit_retries`` is negative.

        """
        if not source:
            raise ValueError("source must not be empty")
        if max_rate_limit_retries < 0:
            raise ValueError("max_rate_limit_retries must be at least zero")

        self.source = source
        self.max_rate_limit_retries = max_rate_limit_retries
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)
        self._owns_client = client is None
        self._next_request_at = 0.0

    def __enter__(self) -> Self:
        """Return this client for use as a context manager."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close the internally managed HTTP client after a context block."""
        if self._owns_client:
            self._client.close()

    def get_documents_with_context_for_meeting(
        self,
        meeting_id: str,
        *,
        ignore_missing_documents: bool = False,
    ) -> list[MeetingDocument]:
        """Fetch every document link from a meeting agenda with its context.

        Agenda items are traversed depth-first in API order. Each document link
        produces one result, including its direct agenda item and all ancestor
        agenda items. Each document link is resolved independently, including
        when a document occurs more than once.

        Args:
            meeting_id: OpenBesluitvorming ID of the meeting.
            ignore_missing_documents: Skip document links whose document entity
                no longer exists. The meeting itself must always exist.

        Returns:
            Document details together with the meeting, document link, and
            agenda-item path for every linked-document occurrence.

        Raises:
            EntityNotFoundError: If the meeting is missing, or a linked
                document is missing while ``ignore_missing_documents`` is
                false.
            ValueError: If the meeting is not a meeting entity or a linked
                document ID refers to a non-document entity.
            httpx.HTTPStatusError: If an API request fails for another reason.
            pydantic.ValidationError: If an API response does not match its model.

        """
        meeting = self._get_entity(meeting_id).to_meeting()
        meeting_documents: list[MeetingDocument] = []
        pending_items: list[PendingAgendaItem] = [
            (agenda_item, (agenda_item,)) for agenda_item in reversed(meeting.agenda or [])
        ]

        while pending_items:
            agenda_item, agenda_item_path = pending_items.pop()
            for document_link in agenda_item.documents or []:
                try:
                    document = self._get_entity(document_link.id).to_document()
                except EntityNotFoundError:
                    if ignore_missing_documents:
                        continue
                    raise

                meeting_documents.append(
                    MeetingDocument(
                        meeting=meeting,
                        agenda_item_path=agenda_item_path,
                        document_link=document_link,
                        document=document,
                    )
                )

            pending_items.extend(
                (child_item, (*agenda_item_path, child_item)) for child_item in reversed(agenda_item.agenda_items or [])
            )

        return meeting_documents

    def get_next_meeting_ids(
        self,
        limit: int,
        *,
        cursor: str = DEFAULT_SNAPSHOT_CURSOR,
    ) -> list[str]:
        """Fetch up to ``limit`` meeting IDs from the snapshot export.

        Args:
            limit: Maximum number of meeting IDs to return.
            cursor: Starting or resume cursor; defaults to ``meeting``. Use
                the last meeting ID from a previous result to resume.

        Returns:
            Meeting IDs in export order. An empty list means no further
            meetings were found. Deleted records and records for other entity
            types are excluded.

        Raises:
            httpx.HTTPStatusError: If a snapshot request fails after retries.
            pydantic.ValidationError: If a record does not match the model.
            ValueError: If the limit, rate-limit headers, or pagination is invalid.

        """
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if not cursor:
            raise ValueError("cursor must not be empty")

        next_cursor = cursor
        seen_cursors = {cursor}
        meeting_ids: list[str] = []
        while True:
            page_limit = min(limit - len(meeting_ids), DEFAULT_EXPORT_LIMIT)
            page = self._get_snapshot_page(cursor=next_cursor, limit=page_limit)
            if len(page.records) > page_limit:
                raise ValueError("Snapshot page exceeds the requested limit")
            meeting_ids.extend(
                record.entity_id for record in page.records if record.op == "upsert" and record.entity_type == "Meeting"
            )

            if not page.has_more:
                return meeting_ids
            next_cursor = page.next_cursor
            if not next_cursor or next_cursor in seen_cursors:
                raise ValueError("Snapshot page has no new continuation cursor")
            if len(meeting_ids) == limit:
                return meeting_ids
            seen_cursors.add(next_cursor)

    def _get_snapshot_page(
        self,
        *,
        cursor: str,
        limit: int = DEFAULT_EXPORT_LIMIT,
    ) -> SnapshotPage:
        """Fetch and validate one NDJSON snapshot page.

        Args:
            cursor: Cursor from which to retrieve the page.
            limit: Maximum number of records requested.

        Returns:
            Validated records and pagination headers.

        """
        params = {"source": self.source, "limit": str(limit), "cursor": cursor}

        response = self._get("/api/export/snapshot", params=params)
        try:
            return SnapshotPage(
                records=[SnapshotRecord.model_validate_json(line) for line in response.iter_lines() if line.strip()],
                next_cursor=response.headers.get("X-Next-Cursor"),
                has_more=response.headers.get("X-Has-More") == "true",
            )
        finally:
            response.close()

    def _get_entity(self, entity_id: str) -> EntityDetail:
        """Fetch and validate one full entity detail response.

        Args:
            entity_id: The entity identifier, which may contain colons.

        Returns:
            The validated entity detail response. API-specific fields not
            declared by the model are preserved.

        Raises:
            EntityNotFoundError: If the entity does not exist.
            httpx.HTTPStatusError: If the API returns an HTTP error.
            pydantic.ValidationError: If the response does not match the model.

        """
        try:
            response = self._get(f"/api/entities/{quote(entity_id, safe='')}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == HTTPStatus.NOT_FOUND:
                exc.response.close()
                raise EntityNotFoundError(f"Entity not found: {entity_id}") from exc
            raise

        try:
            return EntityDetail.model_validate(response.json())
        finally:
            response.close()

    def _get(
        self,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Send a GET request and retry rate-limited responses.

        Args:
            path: Request path relative to the configured base URL.
            params: Optional query parameters.

        Returns:
            The successful HTTP response. The caller owns the response and
            must close it after consuming it.

        Raises:
            httpx.HTTPStatusError: If the response is not successful after
                rate-limit retries.
            ValueError: If a rate-limit response has an invalid or missing
                ``Retry-After`` header.

        """
        for attempt in range(self.max_rate_limit_retries + 1):
            self._wait_for_rate_limit_capacity()
            response = self._client.get(path, params=params)

            if response.status_code != HTTPStatus.TOO_MANY_REQUESTS:
                response.raise_for_status()
                self._schedule_rate_limit_wait(response)

                return response

            if attempt == self.max_rate_limit_retries:
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After")

            if retry_after is None:
                response.close()
                raise ValueError("Rate-limit response has no Retry-After header")

            try:
                delay = float(retry_after)
            except ValueError as exc:
                response.close()

                raise ValueError("Retry-After must be a number of seconds") from exc

            if delay < 0:
                response.close()

                raise ValueError("Retry-After must not be negative")

            response.close()
            sleep(delay)

        raise RuntimeError("Unreachable rate-limit retry state")

    def _wait_for_rate_limit_capacity(self) -> None:
        """Wait until a previously exhausted rate-limit bucket has refilled."""
        delay = self._next_request_at - monotonic()

        if delay > 0:
            sleep(delay)

        self._next_request_at = 0.0

    def _schedule_rate_limit_wait(self, response: httpx.Response) -> None:
        """Schedule a pause when a response exhausts the rate-limit bucket.

        Args:
            response: Successful API response containing rate-limit headers.

        Raises:
            ValueError: If supplied rate-limit headers are not numeric or are
                negative.

        """
        remaining = response.headers.get("RateLimit-Remaining")
        reset = response.headers.get("RateLimit-Reset")

        if remaining is None or reset is None:
            return
        if float(remaining) >= RATE_LIMIT_REQUEST_COST:
            return

        self._next_request_at = max(self._next_request_at, monotonic() + float(reset))
