# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Song matching via bag-of-words hypervectors.

A fun, readable demo of HDC's symbolic, transparent representation style.
Each song is a bundle of word hypervectors — one HV per distinct lyric word,
drawn from a shared codebook — and two songs are "in the same ballpark" when
their bundles are close in cosine similarity.

The headline property: you can *see* why two songs match. The codebook has one
vector per word; the song HV is the normalised sum of those vectors; the
overlap of shared words drives the similarity. No hidden layers, no learned
features, no backprop. Add a new song = bundle its words. Add a new word =
add one row to the codebook.

The bundle-then-cosine pattern is the same single-document context-vector
operation BEAGLE (Jones & Mewhort 2007, *Psychological Review* 114(1): 1-37)
uses to form per-sentence context vectors before accumulating them into
per-word memory; this example stops at the first step. The general
multiset / bag construction is canonised in Kanerva (2009),
*Hyperdimensional Computing: An Introduction*, Cognitive Computation
1(2): 139-159.

Run::

    python examples/song_matching.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc import MAP, RandomEncoder

DIMS = 10_000
SEED = 42

# Eight pseudo-songs grouped into four themes by word overlap.
# No real lyrics; each "song" is a small bag of theme-defining words.
SONGS: dict[str, list[str]] = {
    "midnight_drive": [
        "road",
        "night",
        "stars",
        "highway",
        "driving",
        "empty",
        "window",
        "moon",
    ],
    "summer_road_trip": [
        "road",
        "sun",
        "summer",
        "highway",
        "friends",
        "music",
        "window",
        "driving",
    ],
    "heartbreak_ballad": [
        "heart",
        "broken",
        "tears",
        "alone",
        "crying",
        "rain",
        "missing",
        "lost",
    ],
    "lost_love": [
        "heart",
        "missing",
        "broken",
        "crying",
        "lost",
        "love",
        "tears",
        "memories",
    ],
    "dance_floor": [
        "dance",
        "floor",
        "beat",
        "music",
        "night",
        "party",
        "lights",
        "wild",
    ],
    "party_anthem": [
        "party",
        "night",
        "dance",
        "music",
        "friends",
        "lights",
        "loud",
        "floor",
    ],
    "lullaby": [
        "sleep",
        "baby",
        "night",
        "quiet",
        "dream",
        "stars",
        "moon",
        "soft",
    ],
    "goodnight_song": [
        "night",
        "sleep",
        "dream",
        "moon",
        "quiet",
        "soft",
        "baby",
        "lullaby",
    ],
}


def main() -> None:
    print("Song matching — bag-of-words hypervectors, same theme → same ballpark\n")
    print(f"Dimensions: {DIMS}")
    print(f"Songs:      {len(SONGS)}")

    # Build the shared vocabulary across all songs.
    vocab = sorted({w for words in SONGS.values() for w in words})
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    print(f"Vocabulary: {len(vocab)} unique words\n")

    # One hypervector per word — a codebook you can literally point at.
    key = jax.random.PRNGKey(SEED)
    vsa = MAP.create(dimensions=DIMS)
    enc = RandomEncoder.create(
        num_features=1,
        num_values=len(vocab),
        dimensions=DIMS,
        vsa_model=vsa,
        key=key,
    )
    codebook = enc.codebook[0]  # shape (|vocab|, D)

    # Encode each song as the normalised sum of its word HVs.
    def encode_song(words: list[str]) -> jax.Array:
        idxs = jnp.asarray([word_to_idx[w] for w in words], dtype=jnp.int32)
        summed = jnp.sum(codebook[idxs], axis=0)
        return summed / (jnp.linalg.norm(summed) + 1e-8)

    song_names = list(SONGS.keys())
    song_hvs = jnp.stack([encode_song(SONGS[n]) for n in song_names])

    # Pairwise similarity matrix — it's just a dot product because we normalised.
    sim_mat = song_hvs @ song_hvs.T

    # ---------------------------------------------------------------- table
    print("Pairwise song similarity (cosine, 1.00 = identical theme):")
    print("=" * 96)
    header = " " * 22 + "".join(f"{n[:10]:>11}" for n in song_names)
    print(header)
    print("-" * 96)
    for i, n in enumerate(song_names):
        row = f"{n:<22}"
        for j in range(len(song_names)):
            s = float(sim_mat[i, j])
            row += f"{s:>11.3f}"
        print(row)
    print("=" * 96)

    # ---------------------------------------------------------------- top-2 matches
    print("\nTop match per song (excluding self):")
    print("-" * 80)
    for i, n in enumerate(song_names):
        sims = sim_mat[i].at[i].set(-1.0)  # mask self
        top = int(jnp.argmax(sims))
        best = song_names[top]
        s = float(sims[top])
        shared = sorted(set(SONGS[n]) & set(SONGS[best]))
        print(f"  {n:<22} → {best:<22}  cos={s:.3f}   shared: {', '.join(shared)}")

    # ---------------------------------------------------------------- new query
    print("\nFree-form query — encode any bag of words and search:")
    print("-" * 80)
    query_words = ["road", "driving", "night", "highway"]
    query_hv = encode_song(query_words)
    sims = (song_hvs @ query_hv).tolist()
    ranked = sorted(zip(song_names, sims, strict=True), key=lambda x: -x[1])
    print(f"  query: {query_words}")
    for name, s in ranked[:3]:
        overlap = sorted(set(query_words) & set(SONGS[name]))
        print(f"    {name:<22}  cos={s:.3f}   shared: {', '.join(overlap)}")

    # ---------------------------------------------------------------- teaching moment
    print("\nWhy this works:")
    print("  - The codebook has ONE hypervector per word. That's the 'weights'.")
    print("  - A song HV is just a normalised sum — inspect it, decompose it, edit it.")
    print("  - Two songs share words → their sums overlap → cosine similarity is high.")
    print("  - Zero training. Zero backprop. Add a new song = bundle its words.")


if __name__ == "__main__":
    main()
