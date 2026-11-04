# Review draftu v1 (paper/draft_v1.tex) — aktualizacja 2026-11-05

## 0. Stan blokad przed wysłaniem

| blokada | stan | commit |
|---|---|---|
| Bibliografia `[?]` | **ZAMKNIĘTA**, szczegóły poniżej | f1e0a0b |
| Figury i liczby headline pod settled + pasmo 0.30 m | **ZAPLANOWANE, NIE URUCHOMIONE.** Prereg amend39, kod, 69 komórek, chain na pod. Uruchomienie wymaga poda (koszt). | 888e57e |
| Polish (ton double-blind, §7, Config G/S, limit stron, statements) | **ZAMKNIĘTE** | f1e0a0b |

**Bibliografia.**
- `[?]` pochodziło z kompilacji bez BibTeX: w repo nie było `.bbl`.
- Teraz w repo jest `paper/draft_v1.bbl` i `paper/latexmkrc`. Samo `pdflatex` pokazuje cytowania, a `latexmk` robi
  pełny przebieg.
- Sprawdzone: 0 `[?]` w PDF, bibtex bez ostrzeżeń.
- Dodane albo zweryfikowane online:
  - Ballotta & Talak: IEEE TVT 74(7), 2025, preprint arXiv 2024.
  - Angelopoulos & Bates: FnT ML 16(4), 2023.
  - Molnar i in.: TCST 31(4):1507–1520.
- Brief podaje kolejność „Huang, Caines, Malhamé 2006”, ale praca z 2006 (CIS) ma autorów Huang, **Malhamé**, Caines.
  Zostawiłem kolejność poprawną.
- Pozostałe wpisy są standardowe. Strony konferencyjne nadal warto sprawdzić z rekordem wydawcy.

**Limit stron (CFP IJCAI'27, 2027.ijcai.org, sprawdzone 2026-11-05).**
- Maksymalnie 9 stron: 7 na treść + do 2 na referencje. Przekroczenie oznacza odrzucenie bez recenzji.
- Ethics statement jest opcjonalny i może stać na stronach referencji.
- Obowiązują wytyczne reproducibility. Sekcja „Reproducibility” w pracy je spełnia.
- Double-blind: zero podziękowań i contribution statement. Supplementary do 50 MB, zanonimizowane.
- Stan draftu: treść razem z Reproducibility i Ethical Statement kończy się na s. 7, referencje sięgają s. 8. 8 ≤ 9: OK.
  Cięcie nie było potrzebne.

## 1. Liczby twarde (nie zależą od metryki albo policzone dokładnie pod settled)

| claim | liczba | źródło |
|---|---|---|
| noise_only bez kolizji do N=200 (Config G) | ever 0.000 [0,0] przy każdym N | review_amend24 (eacfb15), review_amend27 (2edfe32) |
| real koliduje od N=50; learned ever 0.000/0.008/0.076/0.473/0.827 | jw. | review_amend22 (c24e62f) |
| dekompozycja τ, część kolizyjna: learned ever przy stałym N = wartość z N-sweepu przy równym T_frame | 0.014/0.072/0.455/0.822 vs 0.008/0.076/0.473/0.827 | review_amend26 (ac32b09) |
| bramka to słaba dźwignia (ever learned 0.07–0.13, real 1.000) | jw. | review_amend25 (1ce2b21) |
| MFG-only = 0.000 (Config G, learned i oracle) | ograniczenie safe ≤ 1−ever jest ciasne i równe 0, więc trzyma pod każdą metryką | review_amend27; amend38 A1_bound |
| warstwy pod settled (Config S, oracle): CBF-clean +0.802 [0.788, 0.818], MFG-clean +0.000 | jw. | review_amend38 A2 (c201490) |
| MFG vs CBF-only (Config S, fresh) +0.318 [0.274, 0.370] | jw. | jw. |
| learned − real (Config S) +0.154 [0.106, 0.208] settled | jw. | jw. |
| delay-free settled 0.920 / 0.800; non-settlery na brzegu (mediana 0.18 m) | jw. | review_amend37/38 |
| open-loop / OOD predyktora (Fig. 2) | v3/real 0.66…0.31; old 5.95 / 25.2 | progress.md, openloop_v3.json |
| G-pred, G-cal, pokrycie w pętli 0.77 / 0.69 | jw. | review_gpred, review_amend31, review_amend35 |
| σ-CBF: werdykt NOT_MET | przy N=200 ever 1.000, więc settled ≤ 0 | review_amend35 |

## 2. Liczby czekające na amend39 (wszystkie @T w drafcie, oznaczone w tekście i podpisach)

| element draftu | co amend39 rozstrzyga (pytanie z preregu) | etap |
|---|---|---|
| Tabela 1 (crossover), Fig. 1, safe_N (real 25, learned 50) | Q1: safe_N pod settled (b=0 główne, b=0.30 pomocnicze). Learned 50 potwierdzone **tylko** przy b=0. | A |
| luka kompensacji +0.936/+0.880/+0.479/+0.137 i „88 %” w abstrakcie | Q2: learned − real safe_settled per N | A |
| mechanizm: noise_only settled-complete w Config G? | Q3 (nowe: dotąd wiemy tylko, że noise_only nie koliduje) | A |
| Fig. 5, „completeness śledzi gęstość” | Q4 | B |
| Fig. 3, wybór marginesu 0.70 | Q5: ta sama reguła z completeness_settled. Jeśli 0.70 wypadnie, zmienia się tekst. | C |
| Fig. 4 | Q6 | C |
| Fig. 6, panele Config G | Q7 | D |

**Czego amend39 nie pokrywa:** pasma 0.30 m dla Config S (Fig. 7 i wiersz Config S w Fig. 6).
- To ma być log-only, ale **pliki trajektorii Config S (`logs_seed*.npz` z amend11/33/34) nie istnieją na tej
  maszynie.** Bieżące repo to świeży klon z 2026-11-04 23:24, a npz nigdy nie były w git.
- amend38 liczył je na innej kopii roboczej.
- **Pytanie do Ciebie:** gdzie jest tamta kopia? Jeśli zniknęła, to Config S też trzeba przeliczyć (CPU, ok. 4 h za
  15 rolloutów według amend36).

## 3. Gdzie claim > dane (aktualne)

1. **NOWE, najważniejsze: prior art Ballotta & Talak (TVT 2025).**
   - Robią predyktor wieku informacji, który przywraca bezpieczeństwo rozproszonego CBF przy opóźnieniu komunikacji,
     w multi-robot collision avoidance.
   - Dla roju płaskiego to zasadniczo nasza „kompensacja przywraca bezpieczeństwo”.
   - W drafcie to już **nie jest wkład**. „How we differ” przeformułowane na:
     - (i) która warstwa w hierarchii MFG+CBF;
     - (ii) staleness vs gęstość przy skalowaniu TDMA z N;
     - (iii) negatyw kalibrowanego marginesu.
   - Recenzent i tak może uznać wkład (2) za oczywisty („CBF jest warstwą bezpieczeństwa, więc kompensacja tam”).
   - Najmocniejsza obrona to wynik oracle: zero dla MFG nawet z idealną estymatą. Warto to wyeksponować w intro.
2. **Tytuł „Staleness, Not Crowding” jest szerszy niż dane.**
   - Trzyma dla kolizji (noise_only, dekompozycja τ).
   - Nie trzyma dla roju bez kompensacji: przy tym samym T_frame liczba sąsiadów ma znaczenie.
   - Completeness zależy od gęstości.
   - Propozycja: „Staleness, Not Density, Limits Safety …”.
3. **safe_N=50 jest zarejestrowane @T.** Punkt N=100 pod settled pochodzi z Config S (inny predyktor, d_safe, seedy), więc
   safe_N **nie jest** zweryfikowane. Rozstrzyga amend39 Q1.
4. **„+0.802 CBF” to oracle, nie predyktor.** Podział warstw z learned pod settled istnieje tylko jako upper bound,
   dopóki amend39 D nie przejdzie.
5. **Rozjazd Config G vs S:** +0.880 vs +0.154. Draft tłumaczy teraz w §3, dlaczego konfiguracje są nieporównywalne
   (geometria, cel, horyzont, margines, seedy, predyktor). Komórki rozdzielającej wpływ predyktora od konfiguracji nadal
   nie ma.
6. **Pasmo 0.30 m to metryka wprowadzona po zobaczeniu wyników (reguła 5).**
   - Zażądana w briefie 2026-11-05, zarejestrowana w amend39 jako *pomocnicza*.
   - Żaden claim nie może stać na samym b=0.30.
   - Twarde jest tylko to, że pasmo nie ratuje bezpieczeństwa opóźnionych (safe ≤ 1−ever).
7. **Mechanizm σ-CBF „concentrated vs broad”** to interpretacja; w drafcie opisana jako „plausible, untested”.
8. **Link 1 niespełniony.** Kryterium „każdy dron dociera” nie spełnia nawet oracle (0.920). Wkłady są sformułowane
   wokół bezpieczeństwa. Ta rama to decyzja PI.
9. **Teza się zmieniła.** Wkład (2) to negacja pierwotnej tezy (CLAUDE.md). Draft jest z tym spójny.
10. **„NN beats classical”** dotyczy tylko cv i smith. cvls (amend32) jest zarejestrowany i nieuruchomiony.

## 4. Plan compute (amend39), do uruchomienia przez Ciebie

- **Pod:** A100-80GB, wolumen ≥ 200 GB, repo na HEAD ≥ 888e57e, wagi v3 na wolumenie (sha sprawdzane w chainie).
- **Start:** `nohup bash logs/overnight/amend39_chain_pod.sh > /workspace/amend39_chain.log 2>&1 &`
  - Bez klucza GitHub: `NO_PUSH=1`.
  - Etapy przez `STAGES="A B"`, jeśli najpierw tylko blokery.
- **Czas:** ok. 4 h (9 solve'ów ok. 1.2 h + 690 rolloutów ok. 2.5 h). Według poprzednich stawek A100 to mniej niż
  $15.
- **Archiwum:** w trakcie albo po wykonaniu na stacji roboczej
  `bash logs/overnight/amend39_pull_archive.sh <host> <port>`. Wolumen poda kasować dopiero po „VERIFIED”.
- **Bramka reprodukcji:** real d0.70 N=100 seed 0 bit-równy amend22. Na innym GPU porównania z rejestrem są tylko opisowe
  (tak stoi w preregu).
- **Po raporcie:**
  - review_amend39 (Q1–Q7);
  - STOP i rozmowa z PI, jeśli zmiana jest na poziomie tezy (learned safe_N < 50 przy b=0 albo noise_only nie
    settled-complete);
  - dopiero potem regeneracja Tabeli 1 i Fig. 1, 3, 4, 5, 6 z `report.json` i zmiana tekstu.
- **Test lokalny:** kod przeszedł tylko na syntetycznych logach (cells, settle, report). Na GPU jeszcze nie działał.
  Pierwszym realnym testem jest bramka reprodukcji.

## 5. Figury

| fig | metryka | status |
|---|---|---|
| 1 crossover | @T (ever niezależne) | → amend39 A |
| 2 open-loop | open-loop | twarda |
| 3 margines | @T | → amend39 C |
| 4 bramka | @T (ever niezależne) | → amend39 C |
| 5 τ-sweep | @T (ever niezależne) | → amend39 B |
| 6 warstwy | G @T + S settled (oracle) | G → amend39 D; S twarda |
| 7 brzeg | settled (S) | twarda; pasmo wymaga npz Config S (§2) |
| 8 σ-CBF | @T | werdykt twardy |

Fig. 1–5 trzeba przy regeneracji przerysować w jednym stylu z 6–8 (`viz/paper_figures.py`).
