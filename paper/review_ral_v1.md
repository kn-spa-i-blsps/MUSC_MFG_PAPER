# Review — paper/ral_v1.tex (IEEE RA-L), 2026-11-05

Zastępuje `draft_v1.tex` (IJCAI'27). Kierunek wg briefu PI z 2026-11-05: najpierw RA-L, IROS później.

## Stan czterech zadań RA-L

| # | zadanie | stan |
|---|---|---|
| 1 | Bibliografia | **ZROBIONE, kompilacja niesprawdzona lokalnie** (szczegóły niżej) |
| 2 | Regeneracja crossover / mechanizm / τ pod settled + pasmo 0.30, 10 seedów, archiwum trajektorii | **ZABLOKOWANE: RunPod 402 (saldo za niskie).** Przygotowane: prereg, addendum 1, generatory tabeli, liczb i Fig. 1. Config S przeliczony lokalnie (addendum 2). |
| 3 | Reformat do RA-L, przycięcie σ-CBF | **ZROBIONE** (`ral_v1.tex`, IEEEtran journal). Szacunek ok. 5 stron z referencjami. Kompilacja niesprawdzona. |
| 4 | Reproducibility statement | **ZROBIONE** (sekcja przed referencjami) |

### 1. Bibliografia
- `refs.bib`:
  - miał błąd składni: nadmiarowy `}` po `angelopoulos2023conformal`. bibtex by się na tym potknął;
  - dodany `shafer2008tutorial` (z briefu: „Vovk/Shafer”);
  - dopisane strony i tomy dla 9 wpisów.
- **Strony wpisane z pamięci, niesprawdzone online.** Pobieranie z sieci było zablokowane. Do sprawdzenia przed camera-ready
  (lista w komentarzu na górze `refs.bib`).
- Statycznie sprawdzone:
  - 23 wpisy, unikalne klucze, nawiasy zbalansowane;
  - każdy `\cite` w `ral_v1.tex` ma wpis;
  - `ledergerber2015robot` jest niecytowany i wypada sam.
- **Brak silnika TeX na tej stacji.** Próba pobrania tectonic została zablokowana. `[?]` znika po przebiegu bibtex
  (`latexmk` / Overleaf, styl `IEEEtran.bst` jest w TeX Live). Nie widziałem PDF-a. Limit stron i overfull boxy trzeba
  sprawdzić po pierwszej kompilacji.

### 2. amend39: co jest gotowe i jak wznowić
- Prereg `4b4ee8e8…` (bez zmian).
- **Addendum 1** `296e700e…`, przed jakimikolwiek danymi: nagłówek pracy to settled 5 s **+ pasmo 0.30 m** (standard PI).
  b=0 raportowane obok. Zarejestrowane odczyty Q1–Q7 bez zmian, refutacja safe_N nadal przy b=0.
- **Addendum 2** `b843808`, log-only, Config S:
  - trajektorie amend11/33/34 **są na tej stacji** (review_draft §2 pytał, gdzie są);
  - b=0 odtwarza amend38 bit w bit.

  | wielkość | b=0 | b=0.30 |
  |---|---|---|
  | fresh (bez opóźnienia) | 0.920 | **1.000** |
  | noise_only | 0.800 | **0.988** |
  | split, czysty CBF | +0.802 | **+0.968** |
  | split, czysty MFG | +0.000 | +0.000 |
  | MFG vs CBF-only | +0.318 | **+0.350** |
  | learned − real | +0.154 | +0.186 |

- **Wznowienie po doładowaniu RunPod.** Pod `sbk37l6nvrg36q` (A100-SXM4-80GB, 250 GB, $1.79/h):
  1. `start`.
  2. Na podzie: `cd /workspace/MUSC_MFG_PAPER && git pull`.
  3. Sprawdzić, że `results/overnight_p1/predictor_v3/predictor_weights.npz` istnieje. **Nie jest w git.** Jeśli brak,
     skopiować ze stacji przez scp, sha 9583cb7c….
  4. `nohup bash logs/overnight/amend39_chain_pod.sh > /workspace/amend39_chain.log 2>&1 &`. Bez klucza GitHub:
     `NO_PUSH=1`.
  5. Czas ok. 4 h, ok. $8 według stawki z list-pods.
  6. W trakcie, na stacji: `bash logs/overnight/amend39_pull_archive.sh <host> <port>`.
- **Po raporcie:**
  1. `review_amend39.md` (Q1–Q7, PLUMBING vs THESIS-CRITICAL).
  2. Jeśli learned safe_N < 50 przy b=0 albo noise_only nie jest settled-complete, to STOP i PI.
  3. `python -m viz.paper_numbers && python -m viz.paper_figures`. Tabela I, Fig. 1 i makra z abstraktu przełączą się
     same, `\ifpending` też.
  4. Ręcznie zostaje: liczby marginesu i bramki (stage C) oraz podział warstw w Config G (stage D). W tekście mają `†`.
- Generatory przetestowane w obu trybach. Tryb legacy odtwarza liczby draftu co do cyfry. Tryb settled sprawdzony na
  **syntetycznym** CSV w scratchpadzie; nic z niego nie weszło do repo.

### 3. Co się zmieniło w treści względem draft_v1
- Format: IEEEtran journal, keywords, `\markboth`, autorzy jako placeholder. RA-L jest single-blind, więc trzeba wpisać
  nazwiska.
- Naprawione błędy draft_v1:
  - pusta sekcja „Settled Arrival and the Rim” została napisana (§V-C, z Fig. rim);
  - sekcja σ-CBF była **zdublowana**, łącznie z dwiema figurami z tym samym `\label`.
- σ-CBF: jeden akapit (§V-D) z werdyktem, mechanizmem i zastrzeżeniem „untested”. Bez figury.
- Fig. 1, 5 i tabela crossover scalone w jedną figurę* (a: N-sweep, b: dekompozycja ever vs T_frame, c: τ). Fig. 3 i 4
  (margines, bramka) zastąpione liczbami w tekście.
- Tytuł: „Staleness, Not Density, Limits Safety…”, zgodnie z uwagą z review_draft §3.2.
- Liczby Config S w tekście są z pasmem, b=0 w nawiasach.

## Do decyzji PI (poziom tezy, reguła 2)
- **Z pasmem 0.30 m link 1 („delay-free works”) praktycznie trzyma:** fresh 1.000, noise_only 0.988 (Config S).
  - Pod b=0 było NOT MET (0.920 / 0.800).
  - To zmiana werdyktu wynikająca z **metryki dodanej po wynikach**. Pasmo zlecił PI, ale reguła 5 każe to nazwać.
    W pracy b=0 stoi obok każdej liczby z pasmem.
  - Czy PI akceptuje taką ramę jako „sukces”?
- Autorzy, afiliacje, URL repo do reproducibility statement.

## Gdzie claim > dane (aktualne)
- Wszystko z `†` jest @T do czasu amend39. Dotyczy to też safe_N=50 dla learned: niezweryfikowane pod settled.
- Ballotta & Talak: wkład „kompensacja przywraca bezpieczeństwo” nie jest nasz. Wkłady sformułowane wokół warstw,
  staleness vs gęstość i negatywu σ.
- „Learned beats classical” dotyczy tylko cv i smith. cvls niewykonany.
- Strony w bib i kompilacja niesprawdzone (patrz wyżej).
