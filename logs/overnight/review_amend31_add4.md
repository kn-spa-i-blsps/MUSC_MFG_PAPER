# REVIEW — amend31 addendum 4 (SUPPLEMENTARY): ablacje, replikacja, zespół chaosu

Raport według kroku 4 z prereg/prereg_projectA_amend31_addendum4.md. Wszystko poniżej jest SUPPLEMENTARY, z wyjątkiem
miejsc, które wprost cytują zarejestrowany werdykt G-win. Raport powstał wyłącznie z wyników zapisanych w repozytorium.
Do jego przygotowania nie uruchomiono żadnego rolloutu, treningu ani rekalibracji. Nie zmieniono żadnego speca, progu,
ramienia ani seeda. Kolejka pozostaje zablokowana, a amend28 nie był uruchamiany.

## Kontrola wstępna

**Kolejność: poprawna.** Addendum 4 zostało zacommitowane, zanim powstał jakikolwiek wynik G-win.
- Commit addendum 4: a87eb0f, 2026-11-02 16:02:01 UTC (17:02:01 +0100).
- Commit zarejestrowanego raportu G-win (report.json): 29e1496, 2026-11-02 17:56:02 UTC.
- `git merge-base --is-ancestor a87eb0f 29e1496` jest prawdziwe, czyli addendum leży w historii raportu.
- `results/amend31_sigma_cbf/report.json` pojawia się po raz pierwszy w 29e1496. Commit e662d89 (run 1, nieudana bramka
  identity) go nie zawiera.
- Przed addendum, w fcc2f7f, były w drzewie pliki metrics dla N=025. Addendum deklaruje, że ich nie otwierano. Tej
  deklaracji nie da się zweryfikować z git. N=025 nie wchodzi do G-win, który obejmuje tylko N=150 i 200.

**SHA256 addendum 4**
- Aktualny plik: `7b8abd149ca7998d0a116f71f7668f194a6268f01a8358fcc18efa964bada48c`.
- Wartość zarejestrowana w prereg/INDEX.md: identyczna.
- Plik w a87eb0f: identyczny.

**HEAD przy generowaniu raportu:** 86f946e9d1e81e7b15898b467330f85cfab5408d. Lokalnie zrobiono fast-forward z
origin/main o 10 commitów poda.

**Zarejestrowany raport G-win już istnieje.** Składa się z `results/amend31_sigma_cbf/report.json` i
`logs/amend31_pod/report.log` (commit 29e1496). Wygenerowała go procedura [Ad1] (`amend31_sigma_cbf report`) w łańcuchu
amend31. Nie generowano go ponownie i nie był zmieniany. Sekcja „G-win” w logs/overnight/review_amend31.md nadal ma
status „(pending)”. To jest review, a nie raport zarejestrowany.

## Kompletność

| zestaw | oczekiwane | faktyczne | uwagi |
|---|---|---|---|
| amend31 zarejestrowany: 5 N × 8 ramion, seedy 0–9 | 400 | 400 | każda z 40 komórek ma 10/10 |
| A: N150 conf_const_a0.05 | 10 | 10 (alias) | d_safe 0.70 jest w f32 równe fixed0.70, więc użyto istniejących rolloutów amend31 bez przeliczania |
| A: N150 fixed_match (d_safe 14.740847) | 10 | 10 | seedy 0–9 |
| A: N200 conf_const_a0.05 (d_safe 0.768591) | 10 | 10 | seedy 0–9 |
| A: N200 fixed_match (d_safe 64.634132) | 10 | 10 | seedy 0–9 |
| B: N150 sigma_bin_a0.05 / best fixed 0.85 / conf_const 0.70 / fixed_match 14.740847 | 4 × 10 | 4 × 10 | seedy 200–209; bez duplikatów f32 |
| B: N200 sigma_bin_a0.05 / best fixed 0.85 / conf_const 0.768591 / fixed_match 64.634132 | 4 × 10 | 4 × 10 | seedy 200–209; bez duplikatów f32 |
| C: 2 ramiona × 2 N × członkowie m=1..5 × seedy 0–9 | 200 (10 na każdego członka) | **0** (0 na każdego z 20 członków) | nie uruchomione, bo bramka g2 nie przeszła |
| bramki g1 / g2 / g3 | 3 | 3 | po 1 rolloucie, seed 0 |

Wszystkie rollouty mają precision=highest i ramię learned. Wyjątek to bramka g1, która używa ramienia real.

Rolloutów faktycznie wykonanych w addendum 4: A 30, B 80, bramki 3, razem 113. Plik
logs/amend31_add4_pod/write_cells.log zaplanował „31 jobs, 310 rollouts” (A 30 + B 80 + C 200). Z tego 200 to C, które
nie zostało uruchomione.

## Bramki (results/amend31_add4/gates.json)

- **g1 PASS.** Harness addendum 4 na fixed 0.70 (real, N=100, seed 0) jest bit-równy amend22 na wszystkich 7
  IDENTITY_KEYS, m.in. collision_pair_steps 46961 = 46961 i completeness 1.0 = 1.0.
- **g2 FAIL.** Rollout z wyłączoną perturbacją miał być bit-równy bazie (sigma_bin_a0.05, N=150, seed 0). Różni się 28
  z 30 logowanych tablic.
  - Pierwszy rozjazd `pos`: krok 20.
  - Baza: collision_pair_steps 1296, min_sep 6.9e-6 m, completeness 0.860.
  - Przebieg z wyłączoną perturbacją: 1588, 0.0029 m, 0.860.
  - Addendum przewidziało tę możliwość: ramię learned na GPU nie jest bit-deterministyczne między uruchomieniami.
    Zgodnie z addendum nie było przeprojektowania.
- **g3 PASS.** Włączona perturbacja zmienia trajektorię.
  - Pierwszy rozjazd `pos`: krok 0.
  - x agenta 0 zmienia się z 4.4533367 na 4.453337, czyli o +1 ulp. Pozostałe wpisy w kroku 0 są równe.

**Konsekwencja zgodna z tabelą bramek:** priorytet C nie został uruchomiony.

# FAKTY ZAREJESTROWANE

Cytat z report.json (29e1496). Ramię główne: sigma_bin_a0.05. Komparator: najlepsze fixed per N według średniego
safe-success na seedach 0–9. Przy N=150 i N=200 było to fixed0.85. CI to seed-level bootstrap 10k, 95%.

| N | Δ safe-success (σ − best) [CI95] | Δ completeness [CI95] | safe_win | completeness NI (−0.02) | per N |
|---|---|---|---|---|---|
| 150 | −0.159 [−0.197, −0.112] | +0.285 [+0.273, +0.297] | false | true | false |
| 200 | −0.188 [−0.200, −0.174] | −0.060 [−0.078, −0.041] | false | false | false |

- **G-win [Ad1]** (safe-success + completeness non-inferiority −0.02): **NOT_MET**.
- **G-win registered v1** (tylko safe-success): **NOT_MET**.
- G-cal: PASS_AFTER_FIX. Szczegóły w review_amend31.md.
- SHA256 aneksu amend31: 12678c58…; addendum 1: 7278e98b…. Oba są w report.json i INDEX.md.

Ten raport nie zmienia ani nie reinterpretuje tego werdyktu.

# WYNIKI SUPPLEMENTARY

## 1. Surowe średnie per ramię i N (seedy 0–9; N=25/50/100 tylko opisowo)

Wartości to safe-success / ever_collided / completeness.

| ramię | N=25 | N=50 | N=100 | N=150 | N=200 |
|---|---|---|---|---|---|
| fixed0.70 | 1.000 / 0.000 / 1.000 | 0.992 / 0.008 / 1.000 | 0.892 / 0.072 / 0.958 | 0.479 / 0.473 / 0.857 | 0.137 / 0.827 / 0.599 |
| fixed0.85 | 1.000 / 0.000 / 1.000 | 1.000 / 0.000 / 1.000 | 0.747 / 0.020 / 0.764 | 0.486 / 0.213 / 0.577 | 0.188 / 0.608 / 0.408 |
| fixed1.00 | 1.000 / 0.000 / 1.000 | 1.000 / 0.000 / 1.000 | 0.686 / 0.010 / 0.692 | 0.411 / 0.099 / 0.453 | 0.175 / 0.483 / 0.330 |
| fixed1.20 | 1.000 / 0.000 / 1.000 | 0.960 / 0.000 / 0.960 | 0.642 / 0.000 / 0.642 | 0.363 / 0.025 / 0.372 | 0.180 / 0.456 / 0.260 |
| sigma_a0.1 | 1.000 / 0.000 / 1.000 | 0.996 / 0.004 / 1.000 | 0.730 / 0.231 / 0.950 | 0.350 / 0.603 / 0.868 | 0.000 / 1.000 / 0.346 |
| sigma_a0.05 | 1.000 / 0.000 / 1.000 | 0.996 / 0.004 / 1.000 | 0.686 / 0.272 / 0.947 | 0.185 / 0.799 / 0.877 | 0.000 / 1.000 / 0.321 |
| sigma_a0.01 | 1.000 / 0.000 / 1.000 | 0.980 / 0.020 / 1.000 | 0.429 / 0.517 / 0.909 | 0.000 / 1.000 / 0.364 | 0.000 / 1.000 / 0.297 |
| sigma_bin_a0.05 | 1.000 / 0.000 / 1.000 | 1.000 / 0.000 / 1.000 | 0.750 / 0.211 / 0.959 | 0.327 / 0.617 / 0.863 | 0.000 / 1.000 / 0.348 |

Ramiona addendum 4 na seedach 0–9:

| ramię | N=150 | N=200 |
|---|---|---|
| conf_const_a0.05 | 0.479 / 0.473 / 0.857 (= fixed0.70) | 0.184 / 0.708 / 0.481 |
| fixed_match | 0.137 / 0.004 / 0.139 | 0.034 / 0.000 / 0.034 |

Bazowe częstości przy N=150 i N=200 to średnie z tabel powyżej.

**Rozdzielczość metryki (reguła 4.1).**
- N=150: safe-success wynosi od 0.000 do 0.486, a completeness od 0.139 do 0.877. Wartości nie są bliskie 0 ani 1 we
  wszystkich ramionach.
- N=200, safe-success: wszystkie cztery ramiona σ mają 0.000, przy ever_collided 1.000. Ramiona fixed i conf_const mają
  od 0.137 do 0.188, a fixed_match 0.034.
  - Metryka nie rozdziela ramion σ między sobą (wszystkie mają 0).
  - Rozdziela σ od ramion stałych.
  - Wszystkie ramiona przy N=200 mają safe-success ≤ 0.188.

## 2. Różnice sigma_bin_a0.05 − komparator przy N=150 i N=200

Metody:
- „unpaired” to diff_ci (ta sama procedura co w raporcie zarejestrowanym).
- „paired” to bootstrap po seedach, 10 000 replikacji.
- p to dokładny test sign-flip: 2^10 = 1024 wzorce, dwustronny. Najmniejsze możliwe p to 2/1024 = 0.0020.
- Holm jest liczony po N∈{150, 200} osobno dla każdej pary komparator × metryka.
- r to korelacja ramion po seedach. Zapis „n/d” oznacza zerową wariancję: σ ma safe-success 0 na wszystkich seedach.

**Seedy 0–9.** Komparator best fixed to fixed0.85, wybrany na tych samych seedach.

| komparator | N | metryka | Δ | unpaired CI95 | paired CI95 | r | p sign-flip | p Holm |
|---|---|---|---|---|---|---|---|---|
| best fixed (0.85) | 150 | safe | −0.159 | [−0.197, −0.112] | [−0.197, −0.113] | 0.033 | 0.0020 | 0.0039 |
| best fixed (0.85) | 150 | compl | +0.285 | [+0.273, +0.297] | [+0.270, +0.299] | −0.370 | 0.0020 | 0.0039 |
| best fixed (0.85) | 200 | safe | −0.188 | [−0.200, −0.174] | [−0.200, −0.174] | n/d | 0.0020 | 0.0039 |
| best fixed (0.85) | 200 | compl | −0.060 | [−0.078, −0.041] | [−0.078, −0.040] | −0.080 | 0.0039 | 0.0039 |
| conf_const_a0.05 | 150 | safe | −0.151 | [−0.201, −0.099] | [−0.200, −0.097] | −0.037 | 0.0039 | 0.0039 |
| conf_const_a0.05 | 150 | compl | +0.005 | [−0.005, +0.015] | [−0.003, +0.013] | 0.392 | 0.3281 | 0.3281 |
| conf_const_a0.05 | 200 | safe | −0.184 | [−0.192, −0.176] | [−0.192, −0.176] | n/d | 0.0020 | 0.0039 |
| conf_const_a0.05 | 200 | compl | −0.132 | [−0.151, −0.113] | [−0.154, −0.110] | −0.514 | 0.0020 | 0.0039 |
| fixed_match | 150 | safe | +0.190 | [+0.156, +0.232] | [+0.153, +0.233] | −0.267 | 0.0020 | 0.0039 |
| fixed_match | 150 | compl | +0.724 | [+0.713, +0.734] | [+0.715, +0.734] | 0.168 | 0.0020 | 0.0039 |
| fixed_match | 200 | safe | −0.034 | [−0.036, −0.031] | [−0.037, −0.031] | n/d | 0.0020 | 0.0039 |
| fixed_match | 200 | compl | +0.315 | [+0.298, +0.333] | [+0.299, +0.333] | 0.081 | 0.0020 | 0.0039 |

**Replikacja, seedy 200–209.** Komparator jest zamrożony. Średnie safe / compl / ever:
- N=150: σ 0.287 / 0.869 / 0.677; best 0.481 / 0.577 / 0.198; conf_const 0.512 / 0.861 / 0.431; fixed_match
  0.130 / 0.131 / 0.003.
- N=200: σ 0.000 / 0.342 / 1.000; best 0.180 / 0.413 / 0.604; conf_const 0.188 / 0.483 / 0.686; fixed_match
  0.034 / 0.034 / 0.000.

| komparator | N | metryka | Δ | unpaired CI95 | paired CI95 | r | p sign-flip | p Holm |
|---|---|---|---|---|---|---|---|---|
| best fixed (0.85) | 150 | safe | −0.193 | [−0.231, −0.157] | [−0.230, −0.160] | 0.146 | 0.0020 | 0.0039 |
| best fixed (0.85) | 150 | compl | +0.292 | [+0.281, +0.303] | [+0.281, +0.304] | −0.103 | 0.0020 | 0.0039 |
| best fixed (0.85) | 200 | safe | −0.180 | [−0.190, −0.171] | [−0.190, −0.171] | n/d | 0.0020 | 0.0039 |
| best fixed (0.85) | 200 | compl | −0.071 | [−0.089, −0.053] | [−0.089, −0.053] | 0.076 | 0.0020 | 0.0039 |
| conf_const_a0.05 | 150 | safe | −0.225 | [−0.266, −0.185] | [−0.250, −0.198] | 0.616 | 0.0020 | 0.0039 |
| conf_const_a0.05 | 150 | compl | +0.007 | [−0.001, +0.015] | [−0.003, +0.018] | −0.538 | 0.2656 | 0.2656 |
| conf_const_a0.05 | 200 | safe | −0.188 | [−0.214, −0.167] | [−0.214, −0.167] | n/d | 0.0020 | 0.0039 |
| conf_const_a0.05 | 200 | compl | −0.141 | [−0.160, −0.122] | [−0.162, −0.121] | −0.143 | 0.0020 | 0.0039 |
| fixed_match | 150 | safe | +0.157 | [+0.123, +0.190] | [+0.125, +0.187] | 0.406 | 0.0020 | 0.0039 |
| fixed_match | 150 | compl | +0.738 | [+0.729, +0.747] | [+0.731, +0.745] | 0.369 | 0.0020 | 0.0039 |
| fixed_match | 200 | safe | −0.034 | [−0.037, −0.030] | [−0.037, −0.030] | n/d | 0.0020 | 0.0039 |
| fixed_match | 200 | compl | +0.309 | [+0.291, +0.326] | [+0.292, +0.326] | 0.137 | 0.0020 | 0.0039 |

**Dodatek: 20 seedów łącznie (0–9 + 200–209).** Nie wchodzi do wniosków. Podane są paired CI95 dla safe-success i
completeness.

| komparator | N=150 safe | N=150 compl | N=200 safe | N=200 compl |
|---|---|---|---|---|
| best fixed | [−0.204, −0.147] | [+0.279, +0.298] | [−0.192, −0.175] | [−0.079, −0.052] |
| conf_const | [−0.219, −0.153] | [−0.000, +0.013] | [−0.199, −0.175] | [−0.152, −0.122] |
| fixed_match | [+0.148, +0.202] | [+0.724, +0.738] | [−0.036, −0.031] | [+0.299, +0.324] |

## 3. Chaos

**Nie obliczono.** Priorytet C (0/200 rolloutów) nie został uruchomiony, bo bramka g2 nie przeszła. Dlatego nie ma:
- SD po członkach 0–5;
- podziału wariancji na część wewnątrz seeda i między seedami;
- różnicy σ − best fixed na średnich po członkach.

W analysis.json te pola mają wartość `null`.

Jedyny zapisany sygnał tego typu pochodzi z g2. Ten sam seed i ta sama konfiguracja, uruchomione powtórnie bez
perturbacji, dały collision_pair_steps 1296 i 1588 (N=150, sigma_bin, seed 0). Pierwszy rozjazd `pos` wystąpił w kroku
20. To jedna para przebiegów, a nie oszacowanie wariancji.

## 4. Pokrycie w pętli zamkniętej (sigma_bin_a0.05, seedy 0–9, CI95 z bootstrapu po seedach)

| N | wariant wierszy | pokrycie | CI95 | offline G-cal TEST (binned α=0.05) | zmiana |
|---|---|---|---|---|---|
| 150 | wszystkie ważne | 0.931 | [0.930, 0.932] | brak referencji offline | — |
| 150 | prawdziwa odległość < 2 m | 0.938 | [0.936, 0.939] | 0.955 | −0.018 |
| 150 | binding (d_safe > 0.70) | 0.903 | [0.900, 0.906] | 0.960 (σ_c > 0.40 m) | −0.057 |
| 200 | wszystkie ważne | 0.634 | [0.632, 0.636] | brak referencji offline | — |
| 200 | prawdziwa odległość < 2 m | 0.701 | [0.698, 0.703] | 0.955 | −0.254 |
| 200 | binding (d_safe > 0.70) | 0.649 | [0.648, 0.650] | 0.955 (σ_c > 0.40 m) | −0.306 |

Koszty opisowe, średnie po seedach 0–9:

| N | ramię | makespan | t do 90 % | średnie ‖u‖ [m/s] | d_safe mean | d_safe p95 | frakcja binding | ever<0.30 |
|---|---|---|---|---|---|---|---|---|
| 150 | sigma_bin_a0.05 | brak (0/10 seedów) | 126.36 s (1/10 seedów) | 1.258 | 14.745 | 0.870 | 0.134 | 0.617 |
| 150 | fixed0.85 | brak (0/10) | brak (0/10) | 1.166 | — | — | — | 0.213 |
| 150 | conf_const (= fixed0.70) | brak (0/10) | brak (0/10) | 1.099 | — | — | — | 0.473 |
| 150 | fixed_match | brak (0/10) | brak (0/10) | 4.581 | — | — | — | 0.004 |
| 200 | sigma_bin_a0.05 | brak (0/10) | brak (0/10) | 5.196 | 64.640 | 6.975 | 0.868 | 1.000 |
| 200 | fixed0.85 | brak (0/10) | brak (0/10) | 1.292 | — | — | — | 0.608 |
| 200 | conf_const | brak (0/10) | brak (0/10) | 1.237 | — | — | — | 0.708 |
| 200 | fixed_match | brak (0/10) | brak (0/10) | 0.169 | — | — | — | 0.000 |

Uwagi do kosztów:
- „Brak” oznacza, że makespan albo t90 był None, bo nie wszyscy agenci dotarli albo nie dotarło 90 % agentów.
- mean_speed_mps to średnia z ‖u‖, czyli prędkości komendowanej (env/rollout.py:262), a nie z przemieszczenia.
- Wartości d_safe istnieją tylko dla ramienia σ. Ramiona stałe mają d_safe równe swojej stałej.
- d_safe mean jest wielokrotnie większy od p95. Średnia jest więc zdominowana przez rzadkie, bardzo duże wartości.
  Przy N=200 seed 0 p50 wynosi 1.24, a p95 7.08.

## 5. Reguły odczytu (punkt 5 kroku 4)

**„σ wygrywa ze stałym harmonogramem marginesu per N”: NIE SPEŁNIONE.** Brak dowodu, że σ per sąsiad przewyższa stały
margines per N.

Reguła wymaga paired LB(safe) > 0 i LB(compl) ≥ −0.02 względem obu komparatorów, przy obu N i w obu zestawach seedów.
Spełniają to tylko dwie z ośmiu kombinacji:

| zestaw | N | vs | LB safe | LB compl | spełnia |
|---|---|---|---|---|---|
| 0–9 | 150 | conf_const | −0.200 | −0.003 | nie |
| 0–9 | 150 | fixed_match | +0.153 | +0.715 | tak |
| 0–9 | 200 | conf_const | −0.192 | −0.154 | nie |
| 0–9 | 200 | fixed_match | −0.037 | +0.299 | nie |
| 200–209 | 150 | conf_const | −0.250 | −0.003 | nie |
| 200–209 | 150 | fixed_match | +0.125 | +0.731 | tak |
| 200–209 | 200 | conf_const | −0.214 | −0.162 | nie |
| 200–209 | 200 | fixed_match | −0.037 | +0.292 | nie |

**„Efekt ponad chaos”: NIE MOŻNA ROZSTRZYGNĄĆ** przy N=150 ani przy N=200. Nie ma danych z priorytetu C, bo g2 nie
przeszła. Twierdzenie o efekcie ponad chaos nie jest stawiane.

**„Kalibracja przenosi się na pętlę zamkniętą”: NIE SPEŁNIONE.** Próg 0.93 musi być osiągnięty na wierszach < 2 m i na
wierszach binding, przy obu N.
- N=150: wiersze < 2 m 0.938 spełniają próg, binding 0.903 nie spełnia.
- N=200: 0.701 i 0.649, oba poniżej progu.

Wniosek jest opisowy i nie zmienia G-win. Nie było rekalibracji ani kolejnej iteracji.

**PARTIAL.** Przy N=150 σ jest lepsze od fixed_match w safe-success (Holm p = 0.0039 w obu zestawach seedów). Przy
N=200 jest gorsze. Według reguł nie jest to potwierdzenie efektu przy N=150. Reguła wymaga obu N i obu komparatorów, a
względem conf_const przy N=150 różnica safe-success jest ujemna (Holm p = 0.0039).

**Stronniczość komparatora.** Best fixed (fixed0.85 przy obu N) został wybrany po runie, na seedach 0–9. Na tych seedach
jest to wybór stronniczy na jego korzyść. Opisano to i nie korygowano.

Opisowo, na zamrożonym komparatorze w replikacji 200–209:
- best fixed ma safe-success 0.481 (N=150) i 0.180 (N=200), wobec 0.486 i 0.188 na seedach 0–9;
- conf_const w replikacji przy N=150 ma 0.512, czyli więcej niż best fixed (0.481).

## 6. Animacje (N=200, seed 0)

Pliki wygenerowano na podzie w commicie 86f946e, z JAX_PLATFORMS=cpu. Mają po 264 klatki.
- results/amend31_add4/anim/N200_seed0_sigma_bin_a0.05.gif
- results/amend31_add4/anim/N200_seed0_best_fixed_fixed0.85.gif
- results/amend31_add4/anim/N200_seed0_fixed_match.gif
- Indeks ścieżek: results/amend31_add4/anim/paths.json

**Animacje NIE są przejrzane wizualnie.** Nie ma ich za takie uważać, dopóki nie obejrzy ich człowiek.

# AWARIE I NAPRAWY

1. **Bramka identity amend31, run 1 (e662d89).** Nie przeszła. Przyczyna to artefakt 1 ulp w f32 d_safe² w trybie σ.
   - Naprawę opisuje addendum 3 (459d624). Run 2 dał identity PASS (6ddfb66).
   - **PLUMBING**, naprawione, nic nie zostało pominięte. Wcześniej opisane w review_amend31.md.
2. **amend28 uruchomiony przez kolejkę po runie 1 (5ff7869).** Zatrzymany w trakcie pierwszego solve'a, tylko logi, bez
   rolloutu. Stało się to przed addendum 4.
   - Po addendum 4 kolejka była zablokowana markerami. amend28, 32, 29 i 30 nie były uruchamiane.
   - **PLUMBING.**
3. **Bramka g2 nie przeszła (8efb71d).** Ramię learned nie jest bit-deterministyczne między uruchomieniami na GPU.
   - Pierwszy rozjazd `pos` w kroku 20; 28 z 30 tablic się różni.
   - Zgodnie z addendum nie było naprawy ani przeprojektowania. Skutek: priorytet C nie został uruchomiony.
   - **PLUMBING** (determinizm infrastruktury). Brak naprawy.
4. **fixed_match ma d_safe 14.740847 m (N=150) i 64.634132 m (N=200), w freeze 815c90b.** To nie jest błąd kodu.
   - Wartość wynika dokładnie z zarejestrowanej definicji z kroku 0b: średnia ważona wierszami po wszystkich ważnych
     wierszach CBF. Zgadza się z kolumną d_mean raportu zarejestrowanego (14.745 i 64.640).
   - Średnia jest zdominowana przez rzadkie, bardzo duże d_safe ramienia σ.
   - Ramię fixed_match ma w efekcie completeness 0.139 i 0.034. Przy N=200 średnie ‖u‖ wynosi 0.169 m/s.
   - **Ograniczenie zarejestrowanej definicji, nie PLUMBING.** Nie poprawiano, bo wymagałoby to zmiany ramienia.

# CZEGO NIE ZROBIONO I DLACZEGO

- **Priorytet C, 0 z 200 rolloutów.** Nie uruchomiono, bo g2 nie przeszła, zgodnie z tabelą bramek. W związku z tym:
  - punkt 3 (chaos) nie został policzony;
  - reguły „efekt ponad chaos” nie można rozstrzygnąć przy N=150 i N=200.
  - Braków nie uzupełniono szacunkami.
- **Bramka g2 nie była powtarzana ani naprawiana.** Addendum 4 tego zabrania („It is not re-engineered”).
- **Nie powtórzono analizy lokalnie.** Logi npz rolloutów nie są w git i nie ma ich lokalnie, bo leżą na wolumenie poda.
  - Raport korzysta z results/amend31_add4/analysis.json, policzonego na podzie z JAX_PLATFORMS=cpu (86f946e).
  - Lokalnie, na CPU i tylko z plików JSON, powstał results/amend31_add4/review_inventory.json: kompletność, bramki i
    wejścia do reguł odczytu.
- **Brak referencji offline dla pokrycia na wszystkich ważnych wierszach.** G-cal TEST raportuje tylko pary < 2 m i
  wiersze binding, więc nie podano zmiany względem offline.
- **Koszty opisowe policzono tylko dla seedów 0–9.** Kod analizy nie liczy ich dla replikacji.
  - makespan jest None w 10/10 seedów każdego ramienia przy N=150 i N=200.
  - t do 90 % ma wartość tylko w 1/10 seedów sigma_bin przy N=150.
- **Korelacja r = n/d tam, gdzie σ ma safe-success 0 na wszystkich seedach** (N=200). Wariancja jest zerowa.
- **Dla wariantu 20 seedów p sign-flip nie zostało policzone** (2^20 wzorców). Ten wariant nie wchodzi do wniosków.
- **Animacje nie zostały obejrzane przez człowieka.**
- **Nie sprawdzano stanu poda ani pliku /workspace/amend31_add4_DONE.** Pełny log łańcucha addendum 4 z dokładnymi
  czasami nie jest w git. Czasy poniżej pochodzą z timestampów commitów.
- **Nie uruchomiono amend28 ani innych zadań kolejki.** Kolejka pozostaje zablokowana.

# Commity, HEAD, SHA, czas, koszt

Commity amend31 i addendum 4, w kolejności, czasy UTC:
- d8bff6f — addendum 1 (prereg)
- f90bea6 — addendum 2 (prereg)
- 4e18618 — run 1, stage 0
- 1349c30 — run 1, stage 1 (G-cal)
- e662d89 — run 1, identity FAIL
- 5ff7869 — amend28 zatrzymany (logi)
- 459d624 — addendum 3 (prereg)
- 6f2d375 — review_amend31, G-cal
- 677421f — run 2, stage 0
- 6ddfb66 — run 2, identity PASS
- fcc2f7f — checkpoint N=025
- **a87eb0f — addendum 4 (prereg), 16:02:01**
- 88afb4e, 4aa0fb8, 52eaf40, a3a530d — checkpointy N=050/100/150/200
- **29e1496 — raport zarejestrowany G-win, 17:56:02**
- 815c90b — add4 step 2: q_raw, freeze, cells (17:56:40)
- 8efb71d — add4 gates: g1 = 1, g2 & g3 = 0 (18:01:24)
- 9395b5e — add4 priorytet A (18:11:20)
- 6e5a6c2 — add4 priorytet B (18:37:06)
- 86f946e — add4 step 4: analysis.json i animacje (19:08:57)

HEAD przed commitem tego raportu: 86f946e9d1e81e7b15898b467330f85cfab5408d. Ten raport jest kolejnym commitem.

SHA256 speców (wszystkie zgodne z prereg/INDEX.md):

| spec | SHA256 |
|---|---|
| aneks amend31 | 12678c58087e2e4c2a5a342f715b76237ff336d7e9b5cc9e4249b04ebd2cc677 |
| addendum 1 | 7278e98b9288a03e1c58c50e9838d369bfb077bb90c569f3356d501c6b031682 |
| addendum 2 | f0297d18b514ea68883d8289f7a8471764b7897e14d156997c8e59c0154a364a |
| addendum 3 | 32a06cbe010e03ba9e9ea3cd4568ecf18d88040b74e74769fc7c7e0ec8c965af |
| addendum 4 | 7b8abd149ca7998d0a116f71f7668f194a6268f01a8358fcc18efa964bada48c |

Czas i koszt, z timestampów commitów, pod sbk37l6nvrg36q, A100-SXM4-80GB, 1.79 USD/h:

| odcinek | od – do | czas | koszt |
|---|---|---|---|
| krok 3 addendum 4 | 18:01:24 – 18:37:06 | 35.7 min | ≈ 1.07 USD |
| kroki 2–4 addendum 4 | 17:56:40 – 19:08:57 | 72.3 min | ≈ 2.16 USD |
| pod od startu | 14:12 – 19:09 | ≈ 4.95 h | ≈ 8.9 USD |

Limit kroku 3 (6 h / 15 USD) nie został osiągnięty. Jeśli pod nadal działa, koszt rośnie. Stanu poda nie sprawdzano.

Pliki JSON:
- results/amend31_add4/analysis.json
- results/amend31_add4/review_inventory.json
- results/amend31_add4/frozen.json
- results/amend31_add4/q_raw.json
- results/amend31_add4/gates.json
- results/amend31_sigma_cbf/report.json
