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

The [hybrid search job](resources/hybrid_search.job.yml) runs the
**`execute_hybrid_search`** task against the synchronized chunks in Lakebase,
combining vector and text search. Its notebook currently runs a fixed example
query and logs results; it is separate from the processing job.

## Status

This is an example project for the LLMOps with Databricks course. It takes
shortcuts for learning purposes, is not finished, and is not production-ready.
The ingestion, embedding, synchronization, and example search code are present,
but the end-user assistant, grounded answers with citations, and
metadata-filtering experience remain goals rather than a completed application.
