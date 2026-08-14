# Task pool

## Provenance

| Input | SHA-256 (first 16) | Rows |
| --- | --- | --- |
| `tasks.v1.jsonl` | `16e128918ee9c868` | 1110 |

| Setting | Value |
| --- | --- |
| tasks | 1110 |
| seed | 42 |
| requested max domain share | 30% |
| effective max domain share | 33.3% |

## By domain

| Domain | Tasks | Share | Within cap |
| --- | --- | --- | --- |
| knowledge | 370 | 33.3% | yes |
| math | 370 | 33.3% | yes |
| reasoning | 370 | 33.3% | yes |

## By difficulty

| Difficulty | Tasks | Share |
| --- | --- | --- |
| easy | 179 | 16.1% |
| hard | 427 | 38.5% |
| medium | 504 | 45.4% |

## By split

| Split | Tasks | Share |
| --- | --- | --- |
| test | 245 | 22.1% |
| train | 648 | 58.4% |
| validation | 217 | 19.5% |

## By source

| Source | Tasks | Licence (as recorded) | Prompt text publishable |
| --- | --- | --- | --- |
| aime2025_i | 15 | MIT | yes |
| aime2025_ii | 13 | MIT | yes |
| arc_challenge | 120 | CC-BY-SA-4.0 | yes |
| gsm8k | 109 | MIT | yes |
| math500 | 233 | MIT | yes |
| mmlu | 184 | MIT | yes |
| mmlu_pro | 186 | CHECK-AT-COLLECTION | yes |
| rlpr | 250 | Apache-2.0 | yes |

## Build losses

| Reason | Rows |
| --- | --- |
| duplicates removed | 0 |
| balance trim: knowledge | 30 |
| balance trim: math | 30 |

## Unavailable sources

| Source | Why |
| --- | --- |
| benchgen_turkish | D:\Ubos\benchgen-router-dataset\data\raw\benchgen_turkish.jsonl not found — export the bundle there before building this source |
| gpqa_diamond | Idavidrein/gpqa (gpqa_diamond/train): Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub. Visit the dataset page at https://huggingface.co/datasets/Idavidrein/gpqa to ask for access. |

Licences marked `CHECK-AT-COLLECTION` are unverified placeholders and block Stage 7.
