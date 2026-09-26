# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Character-trigram language classification with held-out calibration.

This illustrative 100-phrase corpus uses random MAP character vectors, cyclic
permutation, binding, and bundling. It is inspired by the n-gram construction
in Joshi, Halseth and Kanerva (2016), not a reproduction of their benchmark.
Ridge classification, temperature scaling and split conformal use separate
splits. No conditional coverage by language/length or general-corpus accuracy
is established by this small hand-written sample.

Run: python examples/language_identification.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bayes_hdc import (
    MAP,
    ConformalClassifier,
    RandomEncoder,
    RegularizedLSClassifier,
    TemperatureCalibrator,
    bind_map,
    permute,
)

DIMS = 4096
SEED = 42

# A tiny, self-contained corpus: 20 phrases per language (common greetings,
# idioms, pangrams). Used only to illustrate a trigram encoder on a small held-out split.
CORPUS: dict[str, list[str]] = {
    "english": [
        "the quick brown fox jumps over the lazy dog",
        "how are you doing today my friend",
        "she sells sea shells by the sea shore",
        "a picture is worth a thousand words",
        "actions speak louder than words",
        "an apple a day keeps the doctor away",
        "good morning and welcome to our meeting",
        "practice makes perfect they always say",
        "time flies when you are having fun",
        "the early bird catches the worm",
        "beauty is in the eye of the beholder",
        "rome was not built in a single day",
        "better late than never as the saying goes",
        "all that glitters is not gold",
        "a bird in the hand is worth two in the bush",
        "when in rome do as the romans do",
        "the pen is mightier than the sword",
        "curiosity killed the cat",
        "every cloud has a silver lining",
        "do not judge a book by its cover",
    ],
    "spanish": [
        "el rapido zorro marron salta sobre el perro perezoso",
        "como estas tu mi buen amigo",
        "ella vende conchas marinas a la orilla del mar",
        "una imagen vale mas que mil palabras",
        "las acciones hablan mas fuerte que las palabras",
        "una manzana al dia mantiene al medico alejado",
        "buenos dias y bienvenidos a nuestra reunion",
        "la practica hace al maestro siempre dicen",
        "el tiempo vuela cuando te diviertes",
        "al que madruga dios le ayuda",
        "la belleza esta en el ojo del que mira",
        "roma no se construyo en un solo dia",
        "mas vale tarde que nunca como dice el dicho",
        "no todo lo que brilla es oro",
        "mas vale pajaro en mano que cien volando",
        "donde fueres haz lo que vieres",
        "la pluma es mas fuerte que la espada",
        "la curiosidad mato al gato",
        "no hay mal que por bien no venga",
        "no juzgues un libro por su portada",
    ],
    "french": [
        "le rapide renard brun saute par dessus le chien paresseux",
        "comment vas tu mon cher ami",
        "elle vend des coquillages au bord de la mer",
        "une image vaut mille mots",
        "les actions parlent plus fort que les mots",
        "une pomme par jour eloigne le medecin",
        "bonjour et bienvenue a notre reunion",
        "c est en forgeant que l on devient forgeron",
        "le temps passe vite quand on s amuse",
        "l avenir appartient a ceux qui se levent tot",
        "la beaute est dans l oeil de celui qui regarde",
        "rome ne s est pas faite en un jour",
        "mieux vaut tard que jamais comme on dit",
        "tout ce qui brille n est pas or",
        "un tiens vaut mieux que deux tu l auras",
        "a rome il faut faire comme les romains",
        "la plume est plus puissante que l epee",
        "la curiosite est un vilain defaut",
        "apres la pluie le beau temps",
        "il ne faut pas juger un livre a sa couverture",
    ],
    "german": [
        "der schnelle braune fuchs springt uber den faulen hund",
        "wie geht es dir mein guter freund",
        "sie verkauft muscheln am meeresufer",
        "ein bild sagt mehr als tausend worte",
        "taten sagen mehr als worte",
        "ein apfel am tag halt den arzt fern",
        "guten morgen und willkommen zu unserer versammlung",
        "ubung macht den meister sagen sie immer",
        "die zeit vergeht wie im flug",
        "der fruhe vogel fangt den wurm",
        "schonheit liegt im auge des betrachters",
        "rom wurde nicht an einem tag erbaut",
        "besser spat als nie wie man so sagt",
        "es ist nicht alles gold was glanzt",
        "der spatz in der hand ist besser als die taube auf dem dach",
        "wenn du in rom bist mach es wie die romer",
        "die feder ist machtiger als das schwert",
        "neugier hat die katze getotet",
        "nach regen kommt sonnenschein",
        "man soll ein buch nicht nach seinem einband beurteilen",
    ],
    "italian": [
        "la veloce volpe marrone salta sopra il cane pigro",
        "come stai tu mio caro amico",
        "ella vende conchiglie sulla riva del mare",
        "una immagine vale piu di mille parole",
        "le azioni parlano piu delle parole",
        "una mela al giorno toglie il medico di torno",
        "buongiorno e benvenuto alla nostra riunione",
        "la pratica rende perfetti come dicono sempre",
        "il tempo vola quando ci si diverte",
        "chi va piano va sano e va lontano",
        "la bellezza e negli occhi di chi guarda",
        "roma non e stata costruita in un giorno",
        "meglio tardi che mai come si dice",
        "non e tutto oro quel che luccica",
        "meglio un uovo oggi che una gallina domani",
        "paese che vai usanza che trovi",
        "la penna e piu potente della spada",
        "la curiosita uccise il gatto",
        "dopo la pioggia viene il sereno",
        "non giudicare un libro dalla copertina",
    ],
}

LANGUAGES = list(CORPUS.keys())
# 26 lowercase letters and space; other characters are discarded.
ALPHABET = "abcdefghijklmnopqrstuvwxyz "
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}


def encode_text(text: str, codebook: jax.Array, dims: int) -> jax.Array:
    """Encode text as a bundle of position-bound character trigrams.

    For each trigram (c_i, c_{i+1}, c_{i+2}) at position i, form

        tri_i = char_hv[c_i] * permute(char_hv[c_{i+1}], 1) *
                permute(char_hv[c_{i+2}], 2)

    then bundle. This is the classical Joshi/Halseth/Kanerva (2016)
    language-ID encoder: local character order is captured via binding
    and permutation, while long-range position is averaged out.
    """
    chars = [c for c in text.lower() if c in CHAR_TO_IDX]
    if len(chars) < 3:
        return jnp.zeros(dims)
    idx = jnp.asarray([CHAR_TO_IDX[c] for c in chars], dtype=jnp.int32)

    def trigram(i: int) -> jax.Array:
        c0 = codebook[idx[i]]
        c1 = permute(codebook[idx[i + 1]], 1)
        c2 = permute(codebook[idx[i + 2]], 2)
        return bind_map(bind_map(c0, c1), c2)

    n_tri = len(chars) - 2
    trigrams = jax.vmap(trigram)(jnp.arange(n_tri))
    summed = jnp.sum(trigrams, axis=0)
    return summed / (jnp.linalg.norm(summed) + 1e-8)


def main() -> None:
    print("Language identification with PVSA — 5 languages × 20 phrases each\n")

    key = jax.random.PRNGKey(SEED)
    k_cb, _ = jax.random.split(key)

    # Character codebook: one hypervector per allowed character.
    vsa = MAP.create(dimensions=DIMS)
    char_cb = RandomEncoder.create(
        num_features=1,
        num_values=len(ALPHABET),
        dimensions=DIMS,
        vsa_model=vsa,
        key=k_cb,
    ).codebook[0]  # shape (|alphabet|, D)

    # A small illustrative corpus, not a representative language benchmark.
    # Randomly partition all rows before fitting any estimator.
    records = [(text, i) for i, lang in enumerate(LANGUAGES) for text in CORPUS[lang]]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(records))
    train_idx, temp_idx, cal_idx, test_idx = np.split(perm, [50, 65, 80])
    all_hvs = jnp.stack([encode_text(text, char_cb, DIMS) for text, _ in records])
    all_labels = jnp.asarray([label for _, label in records], dtype=jnp.int32)
    train_hvs, train_labels = all_hvs[train_idx], all_labels[train_idx]
    temp_hvs, temp_labels = all_hvs[temp_idx], all_labels[temp_idx]
    cal_hvs, cal_labels = all_hvs[cal_idx], all_labels[cal_idx]
    test_hvs, test_labels = all_hvs[test_idx], all_labels[test_idx]
    test_texts = [records[i][0] for i in test_idx]

    # Fit ridge regression on training hypervectors.
    clf = RegularizedLSClassifier.create(
        dimensions=DIMS,
        num_classes=len(LANGUAGES),
        reg=1.0,
    ).fit(train_hvs, train_labels)
    logits_cal = cal_hvs @ clf.weights
    logits_test = test_hvs @ clf.weights

    # Temperature uses a disjoint split, then freezes before conformal fitting.
    calibrator = TemperatureCalibrator.create().fit(
        temp_hvs @ clf.weights,
        temp_labels,
        max_iters=200,
    )
    probs_cal = calibrator.calibrate(logits_cal)
    probs_test = calibrator.calibrate(logits_test)

    # Conformal wrap at α = 0.2 (target ≥ 80 % marginal coverage).
    # The finite calibration sample can yield large or uninformative sets.
    alpha = 0.2
    conformal = ConformalClassifier.create(alpha=alpha).fit(probs_cal, cal_labels)
    set_mask = conformal.predict_set(probs_test)

    preds = np.asarray(jnp.argmax(probs_test, axis=-1))
    test_labels_np = np.asarray(test_labels)
    test_accuracy = float(np.mean(preds == test_labels_np))
    coverage = float(conformal.coverage(probs_test, test_labels))
    mean_set_size = float(conformal.set_size(probs_test))

    print(f"Test accuracy:             {test_accuracy:.3f}")
    print(f"Conformal coverage @ α={alpha}: {coverage:.3f}  (target ≥ {1 - alpha:.2f})")
    print(f"Mean prediction-set size:   {mean_set_size:.2f}  (of 5 classes)")
    print(f"Fitted temperature T:       {float(calibrator.temperature):.4f}\n")

    print("Per-sample output (first 10 test phrases):")
    print("=" * 72)
    max_probs = np.asarray(jnp.max(probs_test, axis=-1))
    for i in range(min(10, len(test_texts))):
        pred_lang = LANGUAGES[preds[i]]
        true_lang = LANGUAGES[test_labels_np[i]]
        in_set = [LANGUAGES[j] for j in np.where(set_mask[i])[0]]
        mark = "✓" if preds[i] == test_labels_np[i] else "✗"
        txt = test_texts[i][:48]
        print(f"  {mark}  [{pred_lang:>8s} @ {max_probs[i]:.2f}] set={in_set}")
        print(f"      text: '{txt}'  (true: {true_lang})")
    print(
        "\nThese hand-picked phrases illustrate the API. Coverage is marginal\n"
        "under exchangeability, not conditional on sentence length or language.\n"
        "This corpus does not establish performance on general language data."
    )


if __name__ == "__main__":
    main()
