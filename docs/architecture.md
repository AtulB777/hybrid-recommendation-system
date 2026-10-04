# Architecture

## 1. Offline vs online (the key boundary)

Training and bulk recommendation generation never run inside the API process. The two halves
communicate only through two durable artifacts: the **model bundle** (joblib file) and the
**`recommendations` table**.

```mermaid
flowchart LR
    subgraph OFFLINE["OFFLINE — batch job (python -m training.train)"]
        direction TB
        CSV[("CSV / source data")] --> ING["Ingestion<br/>validate · clean · dedupe"]
        ING --> DB1[("PostgreSQL<br/>users · items · interactions")]
        DB1 --> FE["Feature engineering<br/>user · item · interaction features<br/>time-decayed user-item matrix"]
        FE --> TUNE["Tune hybrid weights<br/>(validation split)"]
        TUNE --> EVAL["Evaluate all models<br/>(held-out test split)"]
        FE --> FIT["Fit final models<br/>on all data"]
        FIT --> ART[["Model bundle<br/>model_bundle.joblib"]]
        FIT --> BATCH["Batch top-K for every user<br/>+ explanations"]
        BATCH --> DB2[("recommendations table")]
        EVAL --> REP["docs/evaluation_results.md"]
    end

    subgraph ONLINE["ONLINE — stateless API (uvicorn app.main:app)"]
        direction TB
        API["FastAPI"] --> SVC["RecommendationService"]
        SVC --> C{{"Cache<br/>(in-memory → Redis)"}}
        SVC --> PRE["Precomputed rows"]
        SVC --> RT["Real-time scoring<br/>(loaded models + live DB history)"]
    end

    ART -. loaded at startup .-> SVC
    DB2 -. read .-> PRE
    DB1 -. live history .-> RT
    U(["Client"]) --> API
```

## 2. Recommendation pipeline (per request / per batch user)

```mermaid
flowchart LR
    A([User]) --> B[Interaction history]
    B --> C["Feature engineering<br/>decayed weight vector"]
    C --> D["Candidate generation<br/>top-100 unseen per model"]
    D --> E1[Popularity]
    D --> E2["Content-based<br/>TF-IDF + cosine"]
    D --> E3["Item-CF<br/>cosine on item columns"]
    D --> E4["User-CF<br/>k nearest users"]
    E1 & E2 & E3 & E4 --> F["Min-max normalise<br/>over candidate union"]
    F --> G["Weighted blend<br/>weights depend on history length"]
    G --> H["Rank + exclude seen"]
    H --> I([Top-K + explanation])
```

## 3. Serving a `GET /recommendations/{user_id}` request

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Cache
    participant DB as PostgreSQL
    participant M as Loaded models

    Client->>API: GET /recommendations/42?k=10
    API->>Cache: get(recs:{version}:42:hybrid:10)
    alt cache hit
        Cache-->>API: payload
        API-->>Client: source = "cache"
    else miss
        API->>DB: newest interaction for user 42?
        alt none newer than training data AND ≥K precomputed rows for this model version
            API->>DB: SELECT recommendations
            DB-->>API: rows (source = "precomputed")
        else fresh activity / cold k / other model
            API->>DB: SELECT user's interactions
            API->>M: score(live history vector)
            M-->>API: top-K + explanations (source = "realtime")
        end
        API->>Cache: set(key, payload, ttl)
        API-->>Client: payload
    end
```

## 4. Database schema

```mermaid
erDiagram
    users ||--o{ interactions : makes
    items ||--o{ interactions : receives
    users ||--o{ recommendations : "is shown"
    items ||--o{ recommendations : "is recommended in"

    users {
        int user_id PK
        string country
        string age_group
        datetime signup_date
    }
    items {
        int item_id PK
        string title
        string genres "pipe separated"
        text description
        int release_year
    }
    interactions {
        int id PK
        int user_id FK
        int item_id FK
        string event_type "view|like|purchase"
        float weight
        datetime timestamp
    }
    recommendations {
        int id PK
        int user_id FK
        int item_id FK
        string model_name
        string model_version
        int rank
        float score
        string reason_type
        text explanation
        datetime generated_at
    }
```

`interactions` has a unique constraint on `(user_id, item_id)`: ingestion collapses repeat events
into the strongest event with the latest timestamp.

## 5. Evaluation protocol

```mermaid
flowchart LR
    I[All interactions] --> S["Per-user temporal split<br/>(users with ≥5 interactions)"]
    S --> TR["TRAIN<br/>earliest ~70%"]
    S --> VA["VALIDATION<br/>next ~10%"]
    S --> TE["TEST<br/>latest ~20%"]
    TR --> F1["fit models"] --> T["tune hybrid weights<br/>on VALIDATION"]
    VA --> T
    T --> W["chosen weights"]
    TR --> F2["fit on TRAIN + VAL"]
    VA --> F2
    W --> F2
    F2 --> M["P@K · R@K · MAP@K · NDCG@K<br/>scored on TEST only"]
    TE --> M
```
