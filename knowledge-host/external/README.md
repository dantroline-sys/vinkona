# external/ — third-party datasets (not included)

The optional importers in `knowledgehost/` read bulk third-party datasets from
this directory. They are large (tens of GB) and carry their own licenses, so
they are **not** part of this repository — download the ones you want and drop
the files here. Nothing in the core ingest → distill → query path requires any
of them.

| File(s) expected here | Dataset | Consumed by | Where to get it |
|---|---|---|---|
| `assertions.csv` | ConceptNet 5 assertions | `knowledgehost/conceptnet.py` | https://github.com/commonsense/conceptnet5/wiki/Downloads |
| `causenet-precision.jsonl` | CauseNet-Precision | `knowledgehost/causenet.py` | https://causenet.org |
| `GLUCOSE_training_data_final.csv` | GLUCOSE | `knowledgehost/glucose.py` | https://github.com/ElementalCognition/glucose |
| `v4_atomic_*.csv` | ATOMIC (v4) | `knowledgehost/atomic.py` | https://allenai.org/data/atomic |

Check each dataset's license before use — several are Creative Commons variants
with attribution and/or share-alike terms, which apply to the *data*, not to
this codebase.

Other bulk sources are fetched by their importers directly rather than from
this directory: Wikipedia arrives as a Kiwix ZIM (`zim_path` in `config.toml`),
and the drug-relation import (`knowledgehost/dmap.py`) pulls from the public
RxNorm/RxClass/MED-RT APIs.

Your own documents (PDFs, EPUBs, books) do **not** go here either — point the
`sources` list in `config.toml` at your document folders and run `./ingest.sh`.
