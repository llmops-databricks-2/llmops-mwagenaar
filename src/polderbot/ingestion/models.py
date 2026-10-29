"""Pydantic models for the OpenBesluitvorming API contract."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

type EntityRef = Annotated[
    str,
    Field(min_length=1, description="A non-empty entity identifier or URI."),
]


class OpenBesluitvormingModel(BaseModel):
    """Base model for OpenBesluitvorming API responses."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class AgendaDocumentLink(OpenBesluitvormingModel):
    """A document link within an agenda item."""

    id: EntityRef
    name: str
    file_name: str | None = None
    content_type: str | None = None
    original_url: str | None = None


class AgendaItem(OpenBesluitvormingModel):
    """An agenda item within a meeting's agenda."""

    id: EntityRef
    parent: EntityRef | None = None
    title: str | None = None
    description: str | None = None
    number: str | None = None
    order: float | None = None
    classification: str | None = None
    is_heading: bool | None = None
    start_date: AwareDatetime | None = None
    end_date: AwareDatetime | None = None
    documents: list[AgendaDocumentLink] | None = None
    agenda_items: list[AgendaItem] | None = Field(default=None, alias="agendaItems")


class EntityDetail(OpenBesluitvormingModel):
    """Common fields returned by ``GET /api/entities/{entity_id}``."""

    entity_id: EntityRef = Field(alias="entityId")
    entity_type: str = Field(alias="entityType")
    entity_type_label: str | None = Field(default=None, alias="entityTypeLabel")
    title: str | None = None
    organization: str | None = None
    date: AwareDatetime | None = Field(
        default=None,
        alias="sortDate",
        description=("For meetings: start of the meeting. For documents: the date the document was last discussed."),
    )

    def to_document(self) -> Document:
        """Convert this detail to a document model when its entity type matches.

        Returns:
            A document detail model with the API fields preserved.

        Raises:
            ValueError: If this entity is not a document.

        """
        if not self.entity_type == "Document":
            raise ValueError("Entity is not of type 'Document'")

        return Document.model_validate(self.model_dump(by_alias=True))

    def to_meeting(self) -> Meeting:
        """Convert this detail to a meeting model when its entity type matches.

        Returns:
            A meeting detail model with the API fields preserved.

        Raises:
            ValueError: If this entity is not a meeting.

        """
        if not self.entity_type == "Meeting":
            raise ValueError("Entity is not of type 'Meeting'")

        return Meeting.model_validate(self.model_dump(by_alias=True))


class Meeting(EntityDetail):
    """A meeting detail response returned by ``GET /api/entities/{entity_id}``."""

    agenda: list[AgendaItem] | None = None
    motions: list[Any] | None = None
    recordings: list[Any] | None = None


class Document(EntityDetail):
    """A document detail response returned by ``GET /api/entities/{entity_id}``."""

    date_modified: AwareDatetime | None = Field(default=None, alias="dateModifiedIso")
    markdown_text: str | None = Field(default=None, alias="markdownText")
    download_url: str | None = Field(default=None, alias="downloadUrl")
    content_type: str | None = Field(default=None, alias="contentType")
    pdf_url: str | None = Field(default=None, alias="pdfUrl")
    pdf_entity_id: EntityRef | None = Field(default=None, alias="pdfEntityId")
    meeting_id: EntityRef | None = Field(default=None, alias="meetingId")


class MeetingDocument(OpenBesluitvormingModel):
    """A document detail together with its location in a meeting agenda."""

    meeting: Meeting
    agenda_item_path: tuple[AgendaItem, ...]
    document_link: AgendaDocumentLink
    document: Document


class SnapshotRecord(OpenBesluitvormingModel):
    """A record from the OpenBesluitvorming snapshot export."""

    seq: int
    op: Literal["upsert", "delete"]
    time: AwareDatetime
    entity_id: EntityRef
    entity_type: str
    source_key: str
    supplier: str
    commit_id: str
    content_hash: str
    schema_version: str
    payload: Any = None


class SnapshotPage(OpenBesluitvormingModel):
    """Snapshot records and pagination information from response headers."""

    records: list[SnapshotRecord]
    next_cursor: str | None = None
    has_more: bool
