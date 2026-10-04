"""Synthetic dataset generator.

No public dataset is downloaded: the sandbox this project was built in has no general internet
access, and a self-contained generator makes the project reproducible offline. The generator
plants a *latent* taste structure (each user prefers 1-2 genres, items have a long-tailed
popularity) and adds noise, so the models have real signal to find but nothing is trivially
separable. The latent preferences are NOT written to disk - models only see interactions.

To use a real dataset (e.g. MovieLens) produce three CSVs with the columns documented in
`app/services/ingestion.py` and point `RECSYS_RAW_DATA_DIR` at them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENRE_VOCAB: dict[str, list[str]] = {
    "sci-fi": ["spaceship", "galaxy", "android", "quantum", "colony", "alien", "orbit", "laser", "cyborg", "wormhole", "starbase", "future"],
    "fantasy": ["dragon", "kingdom", "sorcerer", "quest", "elf", "enchanted", "sword", "prophecy", "realm", "wizard", "dungeon", "legend"],
    "romance": ["love", "wedding", "heartbreak", "kiss", "courtship", "letters", "destiny", "soulmate", "passion", "affair", "engagement", "tender"],
    "thriller": ["conspiracy", "detective", "heist", "hostage", "assassin", "chase", "betrayal", "suspect", "manhunt", "ransom", "spy", "cipher"],
    "comedy": ["hilarious", "prank", "awkward", "roommates", "mishap", "sitcom", "slapstick", "banter", "chaos", "wacky", "parody", "quirky"],
    "drama": ["family", "struggle", "courtroom", "redemption", "grief", "ambition", "immigrant", "inheritance", "reunion", "sacrifice", "hardship", "secrets"],
    "documentary": ["investigation", "history", "wildlife", "expedition", "interviews", "archive", "science", "ocean", "civilization", "footage", "explores", "reality"],
    "animation": ["cartoon", "adventure", "animals", "friendship", "magical", "colorful", "toy", "puppy", "rainbow", "musical", "village", "playful"],
}
SHARED_WORDS = ["story", "journey", "world", "life", "unexpected", "city", "night", "secret", "last", "first"]
ADJECTIVES = ["Silent", "Broken", "Golden", "Hidden", "Last", "Midnight", "Crimson", "Lost", "Distant", "Final", "Rising", "Fallen", "Electric", "Hollow", "Wild", "Burning"]
EVENT_WEIGHTS = {"view": 1.0, "like": 2.0, "purchase": 4.0}
COUNTRIES = ["IN", "US", "GB", "DE", "BR", "JP"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]


def generate_synthetic_dataset(
    n_users: int = 500,
    n_items: int = 400,
    n_cold_users: int = 15,
    n_cold_items: int = 20,
    seed: int = 42,
    end_date: str = "2026-09-30",
    window_days: int = 180,
) -> dict[str, pd.DataFrame]:
    """Return {"users", "items", "interactions"} DataFrames.

    `n_cold_users` users and `n_cold_items` items have zero interactions on purpose so the
    cold-start paths of the system are exercised.
    """
    rng = np.random.default_rng(seed)
    genres = list(GENRE_VOCAB)
    n_genres = len(genres)
    end = pd.Timestamp(end_date)
    start = end - pd.Timedelta(days=window_days)

    # ---- items -----------------------------------------------------------------------
    item_rows, used_titles = [], set()
    multi_hot = np.zeros((n_items, n_genres))
    for i in range(n_items):
        item_id = i + 1
        primary = int(rng.integers(n_genres))
        item_genres = [primary]
        if rng.random() < 0.4:
            second = int(rng.choice([g for g in range(n_genres) if g != primary]))
            item_genres.append(second)
        for g in item_genres:
            multi_hot[i, g] = 1.0
        words = []
        for _ in range(12):
            r = rng.random()
            if r < 0.65:
                pool = GENRE_VOCAB[genres[primary]]
            elif r < 0.85 and len(item_genres) > 1:
                pool = GENRE_VOCAB[genres[item_genres[1]]]
            else:
                pool = SHARED_WORDS
            words.append(pool[int(rng.integers(len(pool)))])
        title = None
        for _ in range(30):
            noun = GENRE_VOCAB[genres[primary]][int(rng.integers(12))].title()
            cand = f"{ADJECTIVES[int(rng.integers(len(ADJECTIVES)))]} {noun}"
            if cand not in used_titles:
                title = cand
                break
        title = title or f"Untitled {item_id}"
        used_titles.add(title)
        item_rows.append(
            {
                "item_id": item_id,
                "title": title,
                "genres": "|".join(genres[g] for g in item_genres),
                "description": " ".join(words),
                "release_year": int(rng.integers(2005, 2027)),
            }
        )
    items = pd.DataFrame(item_rows)

    item_pop = rng.lognormal(0.0, 1.0, n_items)  # long-tailed popularity
    cold_item_idx = rng.choice(n_items, size=n_cold_items, replace=False)
    item_pop[cold_item_idx] = 0.0
    warm_idx = np.flatnonzero(item_pop > 0)

    # ---- users -----------------------------------------------------------------------
    prefs = np.full((n_users, n_genres), 0.05)
    for u in range(n_users):
        n_fav = 1 if rng.random() < 0.5 else 2
        favs = rng.choice(n_genres, size=n_fav, replace=False)
        prefs[u, favs] += rng.uniform(0.6, 1.0, size=n_fav)
    users = pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "country": rng.choice(COUNTRIES, size=n_users),
            "age_group": rng.choice(AGE_GROUPS, size=n_users),
            "signup_date": start - pd.to_timedelta(rng.integers(1, 700, size=n_users), unit="D"),
        }
    )
    cold_user_idx = set(rng.choice(n_users, size=n_cold_users, replace=False).tolist())

    # ---- interactions ----------------------------------------------------------------
    rows = []
    span_seconds = int((end - start).total_seconds())
    for u in range(n_users):
        if u in cold_user_idx:
            continue
        # affinity of this user to each item, scaled to (0, 1]
        aff = (multi_hot @ prefs[u]) / multi_hot.sum(axis=1)
        aff = aff / aff.max()
        p = (item_pop[warm_idx] ** 0.6) * (1.0 + 8.0 * aff[warm_idx])
        p = p / p.sum()
        n = int(np.clip(rng.lognormal(2.8, 0.6), 5, 120))
        n = min(n, len(warm_idx))
        chosen = rng.choice(warm_idx, size=n, replace=False, p=p)
        ts = start + pd.to_timedelta(np.sort(rng.integers(0, span_seconds, size=n)), unit="s")
        a = aff[chosen]
        r = rng.random(n)
        p_buy, p_like = 0.05 + 0.25 * a, 0.15 + 0.25 * a
        events = np.where(r < p_buy, "purchase", np.where(r < p_buy + p_like, "like", "view"))
        for item_idx, t, ev in zip(chosen, ts, events):
            rows.append(
                {
                    "user_id": u + 1,
                    "item_id": int(item_idx) + 1,
                    "event_type": str(ev),
                    "weight": EVENT_WEIGHTS[str(ev)],
                    "timestamp": t,
                }
            )
    interactions = pd.DataFrame(rows)
    return {"users": users, "items": items, "interactions": interactions}
