# Retrieval Evaluation Results

Corpus: 16 real indexed documents | 12 labeled queries | k=5

| Config | recall@5 | nDCG@5 | MRR |
|---|---|---|---|
| bm25 | 0.7014 | 0.6156 | 0.6764 |
| dense | 0.8819 | 0.7227 | 0.6667 |
| hybrid | 0.8194 | 0.7198 | 0.7361 |
| hybrid+rerank | 0.8056 | 0.7021 | 0.7153 |

## Per-query detail
```json
[
 {
  "config": "bm25",
  "query": "why did the model predict a titanic passenger would survive",
  "recall": 0.5,
  "ndcg": 0.23719771276929622,
  "retrieved": [
   "0febce0468b5ef8d1e6a707b4cc43201",
   "e684e0a1195779b186b952a16f70f4b3",
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "da18c69f8b1c1d567e446d7a631d49ea"
  ]
 },
 {
  "config": "bm25",
  "query": "what factors made the model predict a passenger did not survive the titanic",
  "recall": 0.5,
  "ndcg": 0.6366824387328317,
  "retrieved": [
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "0febce0468b5ef8d1e6a707b4cc43201",
   "e684e0a1195779b186b952a16f70f4b3",
   "7f53ba41efad12a2cb6474cf18f9d048",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b"
  ]
 },
 {
  "config": "bm25",
  "query": "overview report of the titanic dataset quality and model performance",
  "recall": 1.0,
  "ndcg": 0.8772153153380493,
  "retrieved": [
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "065d828b6fc63de75be29dcdfcfb910f",
   "37ab1fcf645fb914c8f9e804c2c54578",
   "e684e0a1195779b186b952a16f70f4b3",
   "44cacf3bdea0df98be6f91c7599a4bbb"
  ]
 },
 {
  "config": "bm25",
  "query": "which flower measurements drove the iris species classification",
  "recall": 0.6666666666666666,
  "ndcg": 0.5307212739772434,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "06ec82d0830e75c69a2fdaea7968598d",
   "3f586351af31ba753c29c575255a0312",
   "95b9e5ff6f4c456cb7ca0c93a3ff35b7",
   "4247b76b091ce5148c75379ac1f0ed2a"
  ]
 },
 {
  "config": "bm25",
  "query": "summary of the iris dataset analysis",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "065d828b6fc63de75be29dcdfcfb910f",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "e684e0a1195779b186b952a16f70f4b3",
   "3f586351af31ba753c29c575255a0312"
  ]
 },
 {
  "config": "bm25",
  "query": "how did the model estimate the age in rings of an abalone",
  "recall": 0.0,
  "ndcg": 0.0,
  "retrieved": [
   "065d828b6fc63de75be29dcdfcfb910f",
   "a119248e10cb4e4fe41497dbcbd78295",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "e684e0a1195779b186b952a16f70f4b3",
   "056c5cf52e0ff7d3377ff3d7b1203f9a"
  ]
 },
 {
  "config": "bm25",
  "query": "abalone dataset overview and data quality",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "065d828b6fc63de75be29dcdfcfb910f",
   "37ab1fcf645fb914c8f9e804c2c54578",
   "e684e0a1195779b186b952a16f70f4b3",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "3f586351af31ba753c29c575255a0312"
  ]
 },
 {
  "config": "bm25",
  "query": "explanation of a prediction driven by the lifeboat column",
  "recall": 0.25,
  "ndcg": 0.16812753627111746,
  "retrieved": [
   "e684e0a1195779b186b952a16f70f4b3",
   "37ab1fcf645fb914c8f9e804c2c54578",
   "065d828b6fc63de75be29dcdfcfb910f",
   "da18c69f8b1c1d567e446d7a631d49ea",
   "4247b76b091ce5148c75379ac1f0ed2a"
  ]
 },
 {
  "config": "bm25",
  "query": "did travelling third class reduce survival chances",
  "recall": 0.5,
  "ndcg": 0.6131471927654584,
  "retrieved": [
   "09fe745289fadc58bf4411e42cd52814",
   "e684e0a1195779b186b952a16f70f4b3",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "056c5cf52e0ff7d3377ff3d7b1203f9a",
   "37ab1fcf645fb914c8f9e804c2c54578"
  ]
 },
 {
  "config": "bm25",
  "query": "petal width importance for classifying iris species",
  "recall": 1.0,
  "ndcg": 0.6934264036172708,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "3f586351af31ba753c29c575255a0312",
   "06ec82d0830e75c69a2fdaea7968598d",
   "4247b76b091ce5148c75379ac1f0ed2a",
   "a119248e10cb4e4fe41497dbcbd78295"
  ]
 },
 {
  "config": "bm25",
  "query": "regression prediction around 9.5 rings for an oyster",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "dee2a9ad77a3dd0e32ad20ffbe7aa808",
   "7f53ba41efad12a2cb6474cf18f9d048",
   "065d828b6fc63de75be29dcdfcfb910f",
   "37ab1fcf645fb914c8f9e804c2c54578",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b"
  ]
 },
 {
  "config": "bm25",
  "query": "high confidence prediction for iris class 2",
  "recall": 1.0,
  "ndcg": 0.6309297535714575,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "a119248e10cb4e4fe41497dbcbd78295",
   "95b9e5ff6f4c456cb7ca0c93a3ff35b7",
   "3f586351af31ba753c29c575255a0312",
   "e684e0a1195779b186b952a16f70f4b3"
  ]
 },
 {
  "config": "dense",
  "query": "why did the model predict a titanic passenger would survive",
  "recall": 0.5,
  "ndcg": 0.3065735963827292,
  "retrieved": [
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "09fe745289fadc58bf4411e42cd52814",
   "da18c69f8b1c1d567e446d7a631d49ea",
   "95b9e5ff6f4c456cb7ca0c93a3ff35b7",
   "0febce0468b5ef8d1e6a707b4cc43201"
  ]
 },
 {
  "config": "dense",
  "query": "what factors made the model predict a passenger did not survive the titanic",
  "recall": 1.0,
  "ndcg": 0.9828920819566879,
  "retrieved": [
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "09fe745289fadc58bf4411e42cd52814",
   "95b9e5ff6f4c456cb7ca0c93a3ff35b7",
   "da18c69f8b1c1d567e446d7a631d49ea",
   "0febce0468b5ef8d1e6a707b4cc43201"
  ]
 },
 {
  "config": "dense",
  "query": "overview report of the titanic dataset quality and model performance",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "e684e0a1195779b186b952a16f70f4b3",
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "065d828b6fc63de75be29dcdfcfb910f",
   "95b9e5ff6f4c456cb7ca0c93a3ff35b7"
  ]
 },
 {
  "config": "dense",
  "query": "which flower measurements drove the iris species classification",
  "recall": 0.6666666666666666,
  "ndcg": 0.5307212739772434,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "06ec82d0830e75c69a2fdaea7968598d",
   "3f586351af31ba753c29c575255a0312",
   "065d828b6fc63de75be29dcdfcfb910f",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b"
  ]
 },
 {
  "config": "dense",
  "query": "summary of the iris dataset analysis",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "e684e0a1195779b186b952a16f70f4b3",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "065d828b6fc63de75be29dcdfcfb910f",
   "7f53ba41efad12a2cb6474cf18f9d048"
  ]
 },
 {
  "config": "dense",
  "query": "how did the model estimate the age in rings of an abalone",
  "recall": 0.6666666666666666,
  "ndcg": 0.5307212739772434,
  "retrieved": [
   "065d828b6fc63de75be29dcdfcfb910f",
   "dee2a9ad77a3dd0e32ad20ffbe7aa808",
   "4247b76b091ce5148c75379ac1f0ed2a",
   "06ec82d0830e75c69a2fdaea7968598d",
   "09fe745289fadc58bf4411e42cd52814"
  ]
 },
 {
  "config": "dense",
  "query": "abalone dataset overview and data quality",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "065d828b6fc63de75be29dcdfcfb910f",
   "e684e0a1195779b186b952a16f70f4b3",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "37ab1fcf645fb914c8f9e804c2c54578",
   "dee2a9ad77a3dd0e32ad20ffbe7aa808"
  ]
 },
 {
  "config": "dense",
  "query": "explanation of a prediction driven by the lifeboat column",
  "recall": 0.75,
  "ndcg": 0.5143371794949736,
  "retrieved": [
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "95b9e5ff6f4c456cb7ca0c93a3ff35b7",
   "09fe745289fadc58bf4411e42cd52814",
   "0febce0468b5ef8d1e6a707b4cc43201",
   "da18c69f8b1c1d567e446d7a631d49ea"
  ]
 },
 {
  "config": "dense",
  "query": "did travelling third class reduce survival chances",
  "recall": 1.0,
  "ndcg": 0.6934264036172708,
  "retrieved": [
   "056c5cf52e0ff7d3377ff3d7b1203f9a",
   "09fe745289fadc58bf4411e42cd52814",
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "da18c69f8b1c1d567e446d7a631d49ea",
   "95b9e5ff6f4c456cb7ca0c93a3ff35b7"
  ]
 },
 {
  "config": "dense",
  "query": "petal width importance for classifying iris species",
  "recall": 1.0,
  "ndcg": 0.6934264036172708,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "06ec82d0830e75c69a2fdaea7968598d",
   "3f586351af31ba753c29c575255a0312",
   "065d828b6fc63de75be29dcdfcfb910f",
   "7f53ba41efad12a2cb6474cf18f9d048"
  ]
 },
 {
  "config": "dense",
  "query": "regression prediction around 9.5 rings for an oyster",
  "recall": 1.0,
  "ndcg": 0.9197207891481876,
  "retrieved": [
   "dee2a9ad77a3dd0e32ad20ffbe7aa808",
   "4247b76b091ce5148c75379ac1f0ed2a",
   "7f53ba41efad12a2cb6474cf18f9d048",
   "da18c69f8b1c1d567e446d7a631d49ea",
   "065d828b6fc63de75be29dcdfcfb910f"
  ]
 },
 {
  "config": "dense",
  "query": "high confidence prediction for iris class 2",
  "recall": 1.0,
  "ndcg": 0.5,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "3f586351af31ba753c29c575255a0312",
   "a119248e10cb4e4fe41497dbcbd78295",
   "06ec82d0830e75c69a2fdaea7968598d",
   "056c5cf52e0ff7d3377ff3d7b1203f9a"
  ]
 },
 {
  "config": "hybrid",
  "query": "why did the model predict a titanic passenger would survive",
  "recall": 0.5,
  "ndcg": 0.3065735963827292,
  "retrieved": [
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "0febce0468b5ef8d1e6a707b4cc43201",
   "da18c69f8b1c1d567e446d7a631d49ea",
   "e684e0a1195779b186b952a16f70f4b3",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b"
  ]
 },
 {
  "config": "hybrid",
  "query": "what factors made the model predict a passenger did not survive the titanic",
  "recall": 0.75,
  "ndcg": 0.8318724637288826,
  "retrieved": [
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "0febce0468b5ef8d1e6a707b4cc43201",
   "09fe745289fadc58bf4411e42cd52814",
   "e684e0a1195779b186b952a16f70f4b3",
   "da18c69f8b1c1d567e446d7a631d49ea"
  ]
 },
 {
  "config": "hybrid",
  "query": "overview report of the titanic dataset quality and model performance",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "e684e0a1195779b186b952a16f70f4b3",
   "065d828b6fc63de75be29dcdfcfb910f",
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "37ab1fcf645fb914c8f9e804c2c54578"
  ]
 },
 {
  "config": "hybrid",
  "query": "which flower measurements drove the iris species classification",
  "recall": 0.6666666666666666,
  "ndcg": 0.5307212739772434,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "06ec82d0830e75c69a2fdaea7968598d",
   "3f586351af31ba753c29c575255a0312",
   "4247b76b091ce5148c75379ac1f0ed2a",
   "e684e0a1195779b186b952a16f70f4b3"
  ]
 },
 {
  "config": "hybrid",
  "query": "summary of the iris dataset analysis",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "e684e0a1195779b186b952a16f70f4b3",
   "065d828b6fc63de75be29dcdfcfb910f",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "3f586351af31ba753c29c575255a0312"
  ]
 },
 {
  "config": "hybrid",
  "query": "how did the model estimate the age in rings of an abalone",
  "recall": 0.6666666666666666,
  "ndcg": 0.4776237035032179,
  "retrieved": [
   "065d828b6fc63de75be29dcdfcfb910f",
   "dee2a9ad77a3dd0e32ad20ffbe7aa808",
   "a119248e10cb4e4fe41497dbcbd78295",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "7f53ba41efad12a2cb6474cf18f9d048"
  ]
 },
 {
  "config": "hybrid",
  "query": "abalone dataset overview and data quality",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "065d828b6fc63de75be29dcdfcfb910f",
   "e684e0a1195779b186b952a16f70f4b3",
   "37ab1fcf645fb914c8f9e804c2c54578",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "a119248e10cb4e4fe41497dbcbd78295"
  ]
 },
 {
  "config": "hybrid",
  "query": "explanation of a prediction driven by the lifeboat column",
  "recall": 0.25,
  "ndcg": 0.24630238874073,
  "retrieved": [
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "da18c69f8b1c1d567e446d7a631d49ea",
   "e684e0a1195779b186b952a16f70f4b3",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "065d828b6fc63de75be29dcdfcfb910f"
  ]
 },
 {
  "config": "hybrid",
  "query": "did travelling third class reduce survival chances",
  "recall": 1.0,
  "ndcg": 0.9197207891481876,
  "retrieved": [
   "09fe745289fadc58bf4411e42cd52814",
   "056c5cf52e0ff7d3377ff3d7b1203f9a",
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "e684e0a1195779b186b952a16f70f4b3",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b"
  ]
 },
 {
  "config": "hybrid",
  "query": "petal width importance for classifying iris species",
  "recall": 1.0,
  "ndcg": 0.6934264036172708,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "06ec82d0830e75c69a2fdaea7968598d",
   "3f586351af31ba753c29c575255a0312",
   "7f53ba41efad12a2cb6474cf18f9d048",
   "a119248e10cb4e4fe41497dbcbd78295"
  ]
 },
 {
  "config": "hybrid",
  "query": "regression prediction around 9.5 rings for an oyster",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "dee2a9ad77a3dd0e32ad20ffbe7aa808",
   "7f53ba41efad12a2cb6474cf18f9d048",
   "065d828b6fc63de75be29dcdfcfb910f",
   "4247b76b091ce5148c75379ac1f0ed2a",
   "a119248e10cb4e4fe41497dbcbd78295"
  ]
 },
 {
  "config": "hybrid",
  "query": "high confidence prediction for iris class 2",
  "recall": 1.0,
  "ndcg": 0.6309297535714575,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "a119248e10cb4e4fe41497dbcbd78295",
   "3f586351af31ba753c29c575255a0312",
   "e684e0a1195779b186b952a16f70f4b3",
   "056c5cf52e0ff7d3377ff3d7b1203f9a"
  ]
 },
 {
  "config": "hybrid+rerank",
  "query": "why did the model predict a titanic passenger would survive",
  "recall": 1.0,
  "ndcg": 0.5012658353418871,
  "retrieved": [
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "e684e0a1195779b186b952a16f70f4b3",
   "09fe745289fadc58bf4411e42cd52814",
   "da18c69f8b1c1d567e446d7a631d49ea",
   "056c5cf52e0ff7d3377ff3d7b1203f9a"
  ]
 },
 {
  "config": "hybrid+rerank",
  "query": "what factors made the model predict a passenger did not survive the titanic",
  "recall": 0.5,
  "ndcg": 0.5855700749881525,
  "retrieved": [
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "e684e0a1195779b186b952a16f70f4b3",
   "09fe745289fadc58bf4411e42cd52814",
   "056c5cf52e0ff7d3377ff3d7b1203f9a",
   "da18c69f8b1c1d567e446d7a631d49ea"
  ]
 },
 {
  "config": "hybrid+rerank",
  "query": "overview report of the titanic dataset quality and model performance",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "e684e0a1195779b186b952a16f70f4b3",
   "065d828b6fc63de75be29dcdfcfb910f",
   "37ab1fcf645fb914c8f9e804c2c54578",
   "44cacf3bdea0df98be6f91c7599a4bbb"
  ]
 },
 {
  "config": "hybrid+rerank",
  "query": "which flower measurements drove the iris species classification",
  "recall": 0.3333333333333333,
  "ndcg": 0.20210734650054757,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "4247b76b091ce5148c75379ac1f0ed2a",
   "065d828b6fc63de75be29dcdfcfb910f",
   "3f586351af31ba753c29c575255a0312",
   "7f53ba41efad12a2cb6474cf18f9d048"
  ]
 },
 {
  "config": "hybrid+rerank",
  "query": "summary of the iris dataset analysis",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "065d828b6fc63de75be29dcdfcfb910f",
   "e684e0a1195779b186b952a16f70f4b3",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "3f586351af31ba753c29c575255a0312"
  ]
 },
 {
  "config": "hybrid+rerank",
  "query": "how did the model estimate the age in rings of an abalone",
  "recall": 0.3333333333333333,
  "ndcg": 0.20210734650054757,
  "retrieved": [
   "065d828b6fc63de75be29dcdfcfb910f",
   "e684e0a1195779b186b952a16f70f4b3",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "dee2a9ad77a3dd0e32ad20ffbe7aa808",
   "09fe745289fadc58bf4411e42cd52814"
  ]
 },
 {
  "config": "hybrid+rerank",
  "query": "abalone dataset overview and data quality",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "065d828b6fc63de75be29dcdfcfb910f",
   "e684e0a1195779b186b952a16f70f4b3",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "37ab1fcf645fb914c8f9e804c2c54578",
   "3f586351af31ba753c29c575255a0312"
  ]
 },
 {
  "config": "hybrid+rerank",
  "query": "explanation of a prediction driven by the lifeboat column",
  "recall": 0.5,
  "ndcg": 0.36331756126716835,
  "retrieved": [
   "e684e0a1195779b186b952a16f70f4b3",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b",
   "da18c69f8b1c1d567e446d7a631d49ea",
   "056c5cf52e0ff7d3377ff3d7b1203f9a",
   "95b9e5ff6f4c456cb7ca0c93a3ff35b7"
  ]
 },
 {
  "config": "hybrid+rerank",
  "query": "did travelling third class reduce survival chances",
  "recall": 1.0,
  "ndcg": 0.8772153153380493,
  "retrieved": [
   "09fe745289fadc58bf4411e42cd52814",
   "056c5cf52e0ff7d3377ff3d7b1203f9a",
   "e684e0a1195779b186b952a16f70f4b3",
   "44cacf3bdea0df98be6f91c7599a4bbb",
   "dbbddaa4ca2f9cbfb4d7a7aa1860aa9b"
  ]
 },
 {
  "config": "hybrid+rerank",
  "query": "petal width importance for classifying iris species",
  "recall": 1.0,
  "ndcg": 0.6934264036172708,
  "retrieved": [
   "37ab1fcf645fb914c8f9e804c2c54578",
   "3f586351af31ba753c29c575255a0312",
   "06ec82d0830e75c69a2fdaea7968598d",
   "a119248e10cb4e4fe41497dbcbd78295",
   "065d828b6fc63de75be29dcdfcfb910f"
  ]
 },
 {
  "config": "hybrid+rerank",
  "query": "regression prediction around 9.5 rings for an oyster",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "dee2a9ad77a3dd0e32ad20ffbe7aa808",
   "7f53ba41efad12a2cb6474cf18f9d048",
   "065d828b6fc63de75be29dcdfcfb910f",
   "4247b76b091ce5148c75379ac1f0ed2a",
   "da18c69f8b1c1d567e446d7a631d49ea"
  ]
 },
 {
  "config": "hybrid+rerank",
  "query": "high confidence prediction for iris class 2",
  "recall": 1.0,
  "ndcg": 1.0,
  "retrieved": [
   "a119248e10cb4e4fe41497dbcbd78295",
   "3f586351af31ba753c29c575255a0312",
   "37ab1fcf645fb914c8f9e804c2c54578",
   "95b9e5ff6f4c456cb7ca0c93a3ff35b7",
   "7f53ba41efad12a2cb6474cf18f9d048"
  ]
 }
]
```