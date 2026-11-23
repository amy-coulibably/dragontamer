"""Affiche la structure des prompts de chaque approche, sans appeler le modèle.

    python show_prompts.py              toutes les approches
    python show_prompts.py b1 a3b       certaines approches
    python show_prompts.py --example    ajoute un exemple réel de {point_de_controle}
    python show_prompts.py --no-system  sans le prompt système

Les données (transcript, points de contrôle, réponses des étapes précédentes) sont remplacées par des
{placeholders} : il ne reste que les consignes. L'approche tourne avec un faux modèle qui répond « {} » ;
les prompts sont donc ceux du vrai run. Deux appels qui ont la même structure ne sont affichés qu'une fois.
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
LEGEND = """{transcript}                 les tours de l'appel, un par ligne (format décrit dans le prompt système)
{point_de_controle}          le bloc d'un point : « ### Qxx — intitulé », réponses autorisées, critères (--example)
{id}                         l'identifiant du point traité (Q01, Q02...)
{reponses_llm1}              JSON des réponses du premier audit : [{id, reponse, citation, tour}]
{reponse_llmN}, {citation_llmN}, {confirme_llm2}, {argumentaire_llm2}
                             la réponse de l'étape N pour ce point, reprise telle quelle
{citation_du_premier_essai}  la citation du premier essai de B1, introuvable dans le transcript"""


def skeleton(user: str, transcript, checkpoints) -> str:
    """Le prompt avec ses données remplacées par des placeholders, les blocs répétés regroupés."""
    user = user.replace(transcript.text, "{transcript}")
    for cp in checkpoints:
        user = user.replace(prompts.checkpoint_block(cp), "{point_de_controle}")
        user = user.replace(f'{{"id": "{cp["id"]}", "citations"', '{"id": "{id}", "citations"')
    user = re.sub(r"(<premier_audit>\n).*?(\n</premier_audit>)", r"\1{reponses_llm1}\2", user, flags=re.S)
    user = re.sub(r"^Audit 1 : .*$", 'Audit 1 : réponse={reponse_llm1} | citation="{citation_llm1}"', user, flags=re.M)
    user = re.sub(r"^Audit 2 : .*$", 'Audit 2 : réponse={reponse_llm2} | citation="{citation_llm2}" | '
                                     'confirme l\'audit 1={confirme_llm2} | argumentaire="{argumentaire_llm2}"', user, flags=re.M)
    out = []      # paragraphes identiques à la suite -> un seul, avec le nombre de répétitions
    for part in re.split(r"\n{2,}", user):
        if out and out[-1][0] == part:
            out[-1][1] += 1
        else:
            out.append([part, 1])
    return "\n\n".join(p if n == 1 else f"{p}\n    … × {n} : un bloc par point de contrôle" for p, n in out)


def main():
    parser = argparse.ArgumentParser(description="Affiche la structure des prompts des approches, sans appeler le modèle.")
    parser.add_argument("approaches", nargs="*", metavar="CLE", help="a1 a2 a3a a3b b1 b2 (défaut : toutes)")
    parser.add_argument("--example", action="store_true", help="ajoute un exemple réel de {point_de_controle}")
    parser.add_argument("--no-system", action="store_true", help="ne pas afficher le prompt système")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    checkpoints, transcript = load_checkpoints(), load_transcripts(1)[0]
    line = "=" * 100

    if not args.no_system:
        print(f"{line}\nPROMPT SYSTÈME · le même pour tous les appels de toutes les approches\n{line}\n{prompts.SYSTEM}")
    for key in args.approaches or list(APPROACHES):
        if key not in APPROACHES:
            parser.error(f"approche inconnue : {key}")
        calls = []
        APPROACHES[key].run(transcript, checkpoints, lambda system, user, temperature=None: calls.append((system, user)) or {})
        print(f"\n{line}\n{key.upper()} · {APPROACHES[key].name} · {len(calls)} appel(s) par audio\n{line}")
        if key == "b1":      # le second essai ne part que sur un OUI à citation introuvable : on le montre à part
            calls.append((prompts.SYSTEM, prompts.checklist_retry(transcript.text, checkpoints[0], "{citation_du_premier_essai}")))
        shapes = {}          # structure -> [nature de l'appel, nombre d'appels qui ont cette structure]
        for system, user in calls:
            if system != prompts.SYSTEM:
                print("\n(attention : un appel de cette approche utilise un autre prompt système)")
            shapes.setdefault(skeleton(user, transcript, checkpoints), [replay._kind(user)[0], 0])[1] += 1
        for text, (kind, n) in shapes.items():
            print(f"\n----- {NAMES[kind]} · {n} appel(s) de cette forme -----\n{text}")

    print(f"\n{line}\nPLACEHOLDERS\n{line}\n{LEGEND}")
    if args.example:
        print(f"\n----- exemple de {{point_de_controle}} -----\n{prompts.checkpoint_block(checkpoints[0])}")


if __name__ == "__main__":
    main()
