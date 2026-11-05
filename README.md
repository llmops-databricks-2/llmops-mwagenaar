# Polderbot

Polderbot is a work-in-progress Databricks project for a public governance
document assistant for Wetterskip Fryslân. It aims to make reliable information
easier to find without manually reviewing large numbers of meeting documents and
attachments.

The intended data source is public OpenBesluitvorming/Notubiz data: meetings,
agendas, agenda items, and related documents. The project stores document text
alongside metadata such as meeting dates and titles, agenda item titles,
document titles, and source URLs.

The initial goal is a small proof of concept, for example around 10 documents,
with document search, metadata filtering, and an agent that retrieves relevant
fragments and answers using only those sources, with citations. Example questions:

- Which documents discussed water safety in 2025?
- Summarise the proposal for agenda item X.
- In which meetings was water-level management discussed?

## Jobs and tasks

The Databricks Asset Bundle in [databricks.yml](databricks.yml) defines
development (`dev`), acceptance (`acc`), and production (`prd`) targets.
The jobs are:

### Document processing

The [document processing job](resources/document_processing.job.yml) runs these
tasks in order:

1. **`ingest`** retrieves meeting documents and their context from
   OpenBesluitvorming and stores the API-provided Markdown and metadata in a
   Delta table. The `meeting_limit` job parameter defaults to one meeting, not
   a fixed number of documents.
2. **`embed_documents`** chunks the ingested documents and generates embeddings,
   storing the chunks in a Delta table.
3. **`sync_document_chunks`** creates or refreshes a Lakebase synced table for
   the chunks and ensures search indexes are available.

### Hybrid search

The [hybrid search job](resources/hybrid_search.job.yml) runs a separate
notebook demonstrating document search, Genie and Lakebase conversation memory.
The bundle supplies the Genie space ID; set the `genie_space_id` widget when
running interactively. The notebook's execution identity needs permission to
read the `databricks_client_id` and `databricks_client_secret` Unity Catalog
secrets in the project schema. Lakebase memory uses those service principal
credentials; the principal needs Lakebase access and permission to create and
use the session messages table.

The document search tool uses the environment's synced chunks table and
supports inclusive, timezone-aware `document_date_from` and `document_date_to`
bounds. These filter on the date a document was last discussed and exclude
undated documents when a bound is supplied.

### Agent

```text
User message + session ID
        |
        v
Load conversation history from Lakebase
        |
        v
Model selects tools or answers directly
        |
        +-- Document content --> hybrid document search
        +-- Structured questions and counts --> Genie through MCP
        +-- Combined questions --> potentially use both
        |
        v
Execute requested tools and save messages and results
        |
        v
Return tool results to the model; repeat within the iteration limit
        |
        v
Save and return the final answer
```

The model chooses tools and arguments; code handles execution, memory and the
iteration limit. Hybrid search retrieves document passages, while Genie answers
structured questions such as document counts.

Initialize `LakebaseMemory` with `ensure_schema()` and reuse a `session_id` to
resume a conversation. Alternatively, pass `history` without the system prompt
to bypass memory, or omit both for a one-off conversation. Do not combine
`history` and `session_id`. The notebook creates new session IDs on each run.

Serialize turns for each session externally; incomplete tool exchanges are not
repaired. Create a new agent when its 20-minute workspace token expires.
Grounded answers with reliable citations remain a goal.

## Status

This is an example project for the LLMOps with Databricks course. It takes
shortcuts for learning purposes, is not finished, and is not production-ready.
The ingestion, embedding, synchronization, and example search code are present,
but the end-user assistant, grounded answers with citations, and
metadata-filtering experience remain goals rather than a completed application.
