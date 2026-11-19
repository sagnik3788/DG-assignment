# Gulf Property Intelligence

Source-grounded English/Arabic assistant over public DarGlobal and Wasalt pages. Explanatory questions use hybrid BM25 and multilingual E5 vector retrieval with OpenRouter generation; Wasalt listing ID, location, price, and bedroom questions use structured records and deterministic filters. Responses link back to captured pages and show capture dates. The snapshot does not provide live listing availability.

## Current data snapshot

- 179 indexed pages: 47 DarGlobal and 132 Wasalt.
- 47 structured Wasalt listing records, combining language variants by listing ID.
- 18 DarGlobal pages were captured by opening ordinary public pages in a browser, including English pages and Arabic pages reached through the site language control. Other indexed pages were already in the corpus. No access controls were bypassed.
- Collection dates are stored per page in `data/documents.jsonl`; source freshness is shown by source on the live page and per citation.

## Run locally

```sh
cp .env.example .env
podman build -t gulf-property-chat .
podman run --rm -p 7860:7860 -v "$PWD/data:/data:Z" --env-file .env gulf-property-chat
```

Set `OPENROUTER_API_KEY` in `.env` (never commit it); `OPENROUTER_MODEL` defaults to `openrouter/free`. Health: `/health`. Chat API: `POST /api/chat` with `{"message":"..."}`. The mounted `/data` directory persists both the page corpus and `listings.jsonl`. A refreshed corpus should be treated as a dated snapshot and redeployed after updating those files.

## Retrieval evaluation

The labeled set has 70 English and Arabic questions, including exact IDs, attribute filters, cross-language queries, unsupported topics, and live-status questions. The current hybrid retriever scores 95% Recall@5 and Recall@10; structured listing ID/filter checks pass 30/30; the labeled unsupported/live-status checks abstain 20/20. Four additional out-of-domain probes also abstain. These are regression checks, not a production-scale quality guarantee. See [`data/retrieval_questions.json`](data/retrieval_questions.json).

Compare BM25, dense, and hybrid retrieval with `python scripts/evaluate_rag_full.py --backend fastembed --model intfloat/multilingual-e5-small`.

## Deployment

The live demo is hosted at `https://gulf-property-intelligence.latterterrapin32.chatgpt.site`. The Sites Worker bundle includes a versioned copy of both JSONL datasets; a corpus update requires rebuilding and deploying a new version. The OpenRouter key is stored as a deployment secret, not in the repository. The portable Docker image was built with Podman and its mounted-corpus health and chat endpoints were smoke-tested.

The Docker image is configured for Hugging Face Docker Spaces (`sdk: docker`, `app_port: 7860`). It bundles the ONNX embedding model and precomputed vectors during image build, so requests do not need to download the model or build the index. Set `OPENROUTER_API_KEY` as a Space secret. The repository's public demo URL is currently the separate Sites-hosted Worker; the Docker image is the Space-compatible package. A Hugging Face Space ID is not recorded here.
