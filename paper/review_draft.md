# Review draftu v1 (paper/draft_v1.tex) — 2026-11-04

Format: IJCAI-27 (`ijcai27.sty`, `named.bst`, skopiowane z FormattingGuidelines-IJCAI-27). Kompiluje się bez błędów
i bez niezdefiniowanych cytowań. PDF ma 7 stron razem z bibliografią, czyli mieści się w limicie 7 + 2. Build nie
jest commitowany. Przy każdej kluczowej liczbie w .tex stoi komentarz `% SRC:` z plikiem i commitem.

Dwie konfiguracje, których draft nigdy nie miesza w jednym porównaniu:
- **Config G** (geometria crossovera amend21/22): N = 25–200, d_safe 0.70, 10 seedów, A100, metryka @T.
  **Surowe trajektorie stracone.**
- **Config S** (amend11/33/34/38): N = 100, d_safe 0.55, 5 seedów, CPU. Trajektorie są, settled liczony dokładnie.

---

## 1. Liczby twarde (nie zależą od metryki albo są policzone pod settled)

| claim w drafcie | liczba | źródło | dlaczego twarde |
|---|---|---|---|
| noise_only bez kolizji do N=200 | ever = 0.000 [0,0] przy każdym N | review_amend24 (eacfb15), review_amend27 (2edfe32) | ever_collided nie zależy od metryki przybycia |
| real koliduje od N=50 | ever 0.716–1.000 | review_amend22 (c24e62f) | jw. |
| learned ever w N-sweepie | 0.000 / 0.008 / 0.076 / 0.473 / 0.827 | review_amend22 | jw. |
| dekompozycja τ: learned ever przy stałym N = wartość N-sweepu przy równym T_frame | 0.014/0.072/0.455/0.822 vs 0.008/0.076/0.473/0.827 | review_amend26 Q3 (ac32b09) | jw. (część o completeness jest @T, patrz §2) |
| gate = słaba dźwignia (ever learned 0.07–0.13, real 1.000 wszędzie) | jw. | review_amend25 (1ce2b21) | jw. |
| MFG-only kompensacja = 0.000 (Config G, learned i oracle) | 0.000 [0,0] | review_amend27 Q3; amend38 report.json A1_bound (upper bound = 0.0) | ograniczenie safe ≤ 1−ever jest tu ciasne i wynosi 0, więc zero trzyma pod każdą metryką |
| warstwy pod settled (Config S, **oracle**) | CBF-clean +0.802 [0.788, 0.818]; MFG-clean +0.000 [0,0]; noise_only +0.800 | review_amend38 A2 (c201490), report.json A2_contrasts | policzone dokładnie z npz |
| MFG vs CBF-only (Config S, fresh) | +0.318 [0.274, 0.370] settled; 0 kolizji | jw. | jw. |
| learned − real (Config S) | +0.154 [0.106, 0.208] settled | jw. | jw. |
| delay-free settled | fresh 0.920, noise_only 0.800 | review_amend37 (814bd2d), amend38 | jw. |
| non-settlery: 40 (8/run), wszystkie ≤ 0.30 m od brzegu, mediana 0.18 m na zewnątrz, klasy 50/30/8/12 % | jw. | review_amend38 B | jw. |
| terminal cost ON gorszy o ~0.10 pod settled → revert | −0.106 [−0.116, −0.096] | review_amend37 | zarejestrowana reguła |
| tabela open-loop (OOD predyktora, v3/real 0.66…0.31; old 5.95 / 25.2) | jw. | progress.md L57–88, openloop_v3.json (5cd4443) | open-loop nie zależy od metryki przybycia |
| G-pred (AUC 0.781–0.863, R²_log 0.346, R²_tail < 0) | jw. | review_gpred (01701ef) | offline, metryka niezmieniona |
| G-cal (pokrycie offline 0.955 przy α=0.05) | jw. | review_amend31 | jw. |
| pokrycie w pętli: 0.77 near / 0.69 binding przy N=200 | jw. | review_amend35, report.json | jw. |

## 2. Liczby, które czekają na regenerację pod settled (w drafcie oznaczone)

| claim | liczba w drafcie (@T) | dlaczego czeka | co da się powiedzieć już teraz |
|---|---|---|---|
| safe_N: real 25, learned 50 (Config G) | @T | trajektorie amend22 stracone | real nie może wzrosnąć ponad 25. Learned 50 trzyma **wtedy i tylko wtedy, gdy** settled completeness przy N=25 i N=50 wynosi ≥ 0.90 (amend38 A1). **Nierozstrzygnięte.** |
| learned − real safe +0.936/+0.880/+0.479/+0.137 | @T | jw. | górne ograniczenia ze 1−ever: 0.992 / 0.924 / 0.527 / 0.173 dla learned. Real ma 0 przy N ≥ 100 pod każdą metryką. |
| „odzyskuje 88 % luki przy N=100” (abstrakt) | @T | jw. | **Pod settled nie zweryfikowane w Config G.** W Config S tylko 0.154 / 0.800 ≈ 19 % (inny predyktor, patrz §3.3). |
| τ-sweep safe 0.986 → 0.178 i completeness stała przy N=100 | @T | trajektorie amend26 stracone, brak per-agent ever w potrzebnej formie | część ever jest twarda (§1). **Completeness**, a więc i „completeness = gęstość”, czeka na regen. |
| warstwy z learned w Config G (CBF-only 0.918/0.503/0.098) | @T | trajektorie amend27 stracone | górne ograniczenia settled: 0.918 / 0.529 / 0.125 |
| tabela marginesów (0.70 optymalne) | @T | amend23 na pod volume | wybór 0.70 może się przesunąć pod settled, bo completeness przy 0.85/1.00 jest już niska |
| σ-CBF G-win (+0.039 / −0.189) | @T | zarejestrowane @T | werdykt NOT_MET i tak trzyma: przy N=200 każdy agent koliduje (ever 1.000), więc settled ≤ 0 |
| Fig. 1 (crossover) | @T | jw. | podpis to zaznacza |

## 3. Gdzie draft albo brief mówi więcej, niż dane pozwalają

Każdy punkt jest albo złagodzony w drafcie, albo wymaga decyzji PI.

1. **„safe_N=50 potwierdzone przy N=100” (brief): nieprawda w tej formie.**
   - Punkt N=100 zweryfikowany pod settled (amend38 A2) pochodzi z **Config S**, a nie z geometrii crossovera.
   - Ma d_safe 0.55, 5 seedów i **inny predyktor**: model N=100-only z amend11, a nie pooled v3.
   - Potwierdza dwie rzeczy: relację learned > real oraz podział warstw.
   - **Nie potwierdza safe_N.** Draft pisze safe_N jako „zarejestrowane @T, pod settled tylko bound, w regeneracji”.
2. **„CBF +0.802” to kompensacja oracle (noise_only), nie predyktor.**
   - Brief podaje +0.802 obok MFG 0.000, jakby to był wynik kompensacji uczonej.
   - W drafcie: tabela warstw, wiersz Config S, jest jawnie podpisana „oracle (noise_only)”.
   - Podziału warstw z learned pod settled nie ma (Config G stracony). Są tylko upper boundy z §2.
3. **Rozjazd wielkości efektu między konfiguracjami.**
   - learned − real: +0.880 (Config G @T, v3) vs **+0.154** (Config S settled, predyktor N=100-only).
   - Draft podaje obie liczby z adnotacją o różnym predyktorze. Recenzent zapyta, która jest „prawdziwa”.
   - Uczciwa odpowiedź: kompensacja dużo pomaga w geometrii G z v3, mało w S ze starym predyktorem. Nie ma
     komórki, która rozdziela wpływ predyktora od wpływu konfiguracji.
4. **Pasmo tolerancji 0.30 m NIE jest policzone.**
   - Brief: „pasmo ratuje no-delay, a NIE opóźnione — dlatego szkoda realna”.
   - Druga połowa („nie ratuje opóźnionych”) jest twarda dla safe-success przez ograniczenie safe ≤ 1−ever: real 0,
     learned ≤ 0.266 w Config S. Draft to pisze.
   - Pierwsza połowa („ratuje no-delay”) jest **niepoliczona**. d(T) non-settlerów ≤ +0.30 m nie implikuje, że będą
     „settled” w pasmie, bo kryterium to 90 % okna, a nie chwila T.
   - Draft pisze pasmo jako „proponowane, do policzenia”.
   - **Flaga z reguły 5:** to nowa metryka po zobaczeniu wyników. Brief PI jej żąda, więc wolno, ale PI powinien
     zatwierdzić ją jawnie, zanim padnie pierwsza liczba (prereg log-only jak amend37/38).
5. **„Nie zatłoczenie” (tytuł i headline) jest szersze niż dane.**
   - Trzyma dla kolizji przy noise_only i dla kompensowanego roju (dekompozycja τ).
   - Dla niekompensowanego roju liczba sąsiadów ma znaczenie przy tym samym T_frame (real ever 1.000 przy N=100 vs
     0.944 przy N=50, T_frame 0.25 s).
   - Completeness zależy od gęstości.
   - Draft precyzuje to w treści („collisions track staleness, completeness tracks density”), ale tytuł jest
     chwytliwy. Do rozważenia: „Staleness, Not Density, Limits Safety …”.
6. **„MFG zarabia na anti-deadlock”.**
   - Porównanie fresh vs fresh_nomfg dotyczy tylko oracle, Config S, 5 seedów.
   - CBF-only to „decoupled single-agent routing” (c_w = w_bar = 0), a nie brak routingu.
   - Wniosek jest więc o sprzężeniu MFG wobec routingu niesprzężonego. Draft tak go opisuje.
   - „Zamrożone” opiera się na niezmienności względem okna (amend37) i na liczbie 34–49 zamrożonych agentów z
     amend36. Ta druga pochodzi z kontrolera z terminal cost ON, dlatego w drafcie jej nie ma.
7. **Mechanizm σ-CBF „concentrated vs broad” to interpretacja, nie wynik.**
   - Liczby są twarde: binding 12.6 % vs 75.2 % wierszy, średnia σ 0.30 vs 0.64 m.
   - Przyczynowość nie była testowana. H1–H5 z review_amend31_diag są NIEROZSTRZYGNIĘTE.
   - Draft pisze to wprost.
8. **σ-CBF +0.039 przy N=150** pochodzi z innego GPU (RTX PRO 6000).
   - Na A100 bez capu wyszło −0.159.
   - Komparatory przepuszczono na tym samym GPU, więc porównanie jest wewnątrz GPU, ale replikacji capu na A100 nie ma.
9. **Link 1 nie jest spełniony.**
   - Kryterium sukcesu PI („każdy dron dociera i nigdy nie koliduje”) nie spełnia żaden wariant, również oracle
     (0.920).
   - Draft mówi to w §6 i w Limitations. Główne wkłady są sformułowane wokół bezpieczeństwa, a nie pełnego sukcesu.
     PI powinien świadomie przyjąć taką ramę.
10. **Zmiana tezy (CLAUDE.md).**
    - Pierwotna teza: kompensacja dla warstwy MFG.
    - Wkład (2) draftu to negacja tej tezy.
    - Draft jest z tym spójny i nie próbuje ratować starej tezy. To jednak decyzja PI o ramie pracy, nie moja.
11. **„NN beats classical”** jest prawdziwe tylko względem cv i smith, które zawodzą z identyfikowalnych powodów.
    cvls (amend32) jest zarejestrowany, nieuruchomiony. Draft to kwalifikuje.
12. **Related work.**
    - Bibliografia (`refs.bib`) jest pisana z pamięci. Każdy wpis trzeba sprawdzić z rekordem wydawcy (strony,
      tomy, rok Molnar 2023 / Elamvazhuthi 2019 vs 2020).
    - Zdanie „prior work … treats the predictor as an input to one layer” to uogólnienie bez przeglądu literatury.
      Do potwierdzenia albo złagodzenia.
    - W repo nie znalazłem listy cytowań prior work z prereg §2.
13. **Brakujące detale modelu.**
    - Horyzont Config G, wagi kosztu MFG (w_goal 2.0, w_hold 8.0, c_w 1.0), siatka 0.2 m, solver.
    - Są w `experiments/amend21_cells/*.yaml`, ale w drafcie ich nie ma. Do dopisania w appendixie albo supplemencie.

## 4. Regeneracja i brakujące liczby

Jedno zdanie celu na pozycję (reguła 1). Nic tu nie jest uruchomione ani zarejestrowane; to wymaga decyzji PI.

| co | koszt | jak służy celowi |
|---|---|---|
| pasmo 0.30 m na istniejących logach Config S | log-only, CPU | czy „delay-free works” pod uczciwą, ale sensowną definicją przybycia (link 1) |
| regen Config G: crossover (real/learned/noise_only), N=25–200, z zapisem trajektorii | GPU | potwierdza albo obala safe_N i „88 %” pod settled (linki 2–3) |
| regen τ-sweep (amend26) | GPU | „completeness = gęstość” pod settled (link 2) |
| regen podziału warstw z learned (amend27 Q3) | GPU | podział warstw z predyktorem, nie tylko z oracle, pod settled (link 4) |
| cvls (amend32, już zarejestrowany) | GPU | czy NN jest potrzebny, czy wystarcza klasyczny estymator (link 3) |

## 5. Werdykt

Draft jest uczciwy wobec danych. Twarde są bezpieczeństwo (ever), mechanizm, podział warstw z oracle i negatyw σ-CBF.
Wielkości sukcesu w Config G są @T i jawnie czekają na regenerację.

Przed wysyłką trzeba jeszcze:
- (i) zregenerować Config G pod settled;
- (ii) policzyć pasmo po zatwierdzeniu go przez PI;
- (iii) zweryfikować bibliografię;
- (iv) zdecydować o tytule (punkt 3.5) i o ramie wobec niespełnionego linku 1 (punkty 3.9–3.10).
