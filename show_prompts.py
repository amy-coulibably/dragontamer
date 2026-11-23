"""Affiche les prompts qu'une approche envoie au modèle, sans appeler le modèle.

    python show_prompts.py                 toutes les approches, premier audio
    python show_prompts.py b1              une approche
    python show_prompts.py a3b --audio 3   quatrième audio
    python show_prompts.py b1 --full       transcript en entier (par défaut : abrégé)
    python show_prompts.py a1 --all        tous les appels (par défaut : le premier de chaque sorte)

L'approche tourne avec un faux modèle qui répond « {} » : les prompts sont ceux du vrai run.
Les prompts qui reprennent une réponse du modèle (LLM2, LLM3, second essai de B1) montrent donc
des réponses vides ou INVALIDE à la place des vraies.
"""
import argparse
import re
import sys

import prompts
import replay
from approaches import APPROACHES
from data import load_checkpoints, load_transcripts

NAMES = {"single": "appel dédié à un point", "batch": "appel groupé", "critic": "LLM2 · critique",
         "judge": "LLM3 · juge", "checklist": "checklist", "retry": "checklist · second essai"}


def main():
    parser = argparse.ArgumentParser(description="Affiche les prompts des approches, sans appeler le modèle.")
    parser.add_argument("approaches", nargs="*", metavar="CLE", help="a1 a2 a3a a3b b1 b2 (défaut : toutes)")
    parser.add_argument("--audio", type=int, default=0, help="rang de l'audio (défaut : 0, le premier)")
    parser.add_argument("--full", action="store_true", help="ne pas abréger le transcript")
    parser.add_argument("--all", action="store_true", help="tous les appels, pas seulement le premier de chaque sorte")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    checkpoints, transcript = load_checkpoints(), load_transcripts()[args.audio]

    for key in args.approaches or list(APPROACHES):
        if key not in APPROACHES:
            parser.error(f"approche inconnue : {key}")
        calls = []
        APPROACHES[key].run(transcript, checkpoints, lambda system, user, temperature=None: calls.append((system, user)) or {})
        if key == "b1":      # le second essai ne se déclenche que sur un OUI à citation introuvable : on le montre à part
            calls.append((prompts.SYSTEM, prompts.checklist_retry(transcript.text, checkpoints[0], "citation du premier essai")))
        print(f"\n{'#' * 100}\n# {key.upper()} · {APPROACHES[key].name} · {transcript.file} · {len(calls)} appel(s)\n{'#' * 100}")
        seen = set()
        for n, (system, user) in enumerate(calls, 1):
            kind = replay._kind(user)[0]
            if kind in seen and not args.all:
                continue
            if not seen:
                print(f"\n===== SYSTEM (identique pour tous les appels sauf mention contraire) =====\n{system}")
            elif system != calls[0][0]:
                print(f"\n===== SYSTEM de l'appel {n} =====\n{system}")
            seen.add(kind)
            if not args.full:
                lines = transcript.text.splitlines()
                if len(lines) > 6:
                    short = "\n".join(lines[:3] + [f"[… {len(lines) - 5} tours masqués, --full pour tout voir …]"] + lines[-2:])
                    user = user.replace(transcript.text, short)
            print(f"\n===== USER · appel {n} sur {len(calls)} · {NAMES[kind]} · {len(user)} caractères affichés =====\n{user}")


if __name__ == "__main__":
    main()
