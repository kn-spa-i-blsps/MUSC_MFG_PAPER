# REVIEW — amend31: diagnoza porażki sigma_bin_a0.05 (tylko odczyt)

Porównanie sigma_bin_a0.05 (σ) z fixed0.70 (fx). N = 100, 150, 200; seedy 0–9.

**Źródła**
- Kod w HEAD e6daf92.
- Istniejące logi npz rolloutów amend31 na wolumenie poda sbk37l6nvrg36q.
- Liczby w results/amend31_add4/diag.json.

**Ograniczenia pracy**
- Analiza: numpy na CPU poda, bez importu JAX. Nie było rolloutów, treningu ani rekalibracji. Nie zmieniono kodu,
  speców, progów ani wyników. Kolejka nadal zablokowana.
- Nie ma tu propozycji naprawy ani oceny, czy metodę da się uratować.

**Najważniejsze ograniczenie danych.** Komórki amend31 mają `log_xhat: false`
(experiments/amend31_cells/N150_sigma_bin_a0.05.yaml:3). Logi zawierają tylko agregaty na krok, sumowane po wierszach
CBF albo po parach:
- histogram d_safe w koszykach 0.01 m; górny koszyk ≥ 10.30 m jest ucięty (env/rollout.py:115);
- sumy d_safe i σ_c oraz liczniki pokrycia;
- MSE x̂ po wszystkich parach i po parach < 1 m.

Per agent zalogowane są: pos, u, u_nom, slack_max, row_violation i intervention. Wartości per wiersz lub per para (σ_c,
d_safe, ‖x̂ − x‖, cechy) nie są zalogowane i są oznaczone „BRAK W LOGACH”.

**Okna czasu.** W1–W5 to pięć równych przedziałów kroków rolloutu, dt = 0.01 s.

## 1. KOD

**a) Limit górny σ_c, g_log, d_safe: brak.**
- g_log: `slog = o[...,0]*lsd + lmu`, liniowe wyjście MLP bez obcięcia (env/rollout.py:167–172).
- σ_c: `sigma_c = q*exp(slog)` (env/rollout.py:69). Mnożnik q należy do zbioru {1.979, 2.104, 2.181, 2.222, 2.307}
  (komórka, margin_q_vals), a exp nie jest ograniczony.
- d_safe: `d_safe = max(margin_floor, 0.30 + margin_k*sigma_c)` (env/rollout.py:70). Komentarz w
  control/cbf.py:55–56 mówi wprost „no cap”.
- margin_d_max (control/cbf.py:50) jest używany tylko w trybie staleness (env/rollout.py:83).
- Jedyne obcięcie w ścieżce to prawa strona wiersza po normalizacji, `clip(lo/|a|, −1e3, 1e3)` (control/cbf.py:29 i
  117). Ono nie ogranicza samego d_safe.
- Przy N=200 średnia d_safe w jednym kroku dochodzi do 2.77e5 m (zob. punkt 2).

**b) Człon od zmiany d_safe w czasie: brak.**
- h_ij = ‖x_i − x̂_ij‖² − d_safe_ij² jest liczone z bieżącego d_safe (control/cbf.py:90).
- Wiersz to 2(x_i − x̂_ij)·u ≥ −a_ij·γ·h_ij (control/cbf.py:5–6, 96–97). Nie ma członu −2·d_safe·ḋ_safe ani członu od
  ruchu x̂_j.

**c) Wiersz, gdy d_safe > d̂ (szacowana odległość).**
- Wtedy h < 0. Udział a_ij jest wymuszony na 0.5 (control/cbf.py:95), a prawa strona lo = −0.5·γ·h > 0.
- Po normalizacji przez |a| = 2d̂ (control/cbf.py:115–117) wymagana prędkość oddalania to
  5·(d_safe² − d̂²)/d̂ m/s (γ = 20), obcięta do 1000.
  - d_safe = 2, d̂ = 1: 15 m/s, więcej niż u_box = 5.66 na oś.
- Wiersz jest ważny tylko, gdy d̂ < r_sense = 5 m (control/cbf.py:83), dla 8 najbliższych według d̂ (control/cbf.py:80).
  Dlatego każdy ważny wiersz z d_safe > 5 m spełnia d_safe > d̂.

**d) Slack i limity u (solver exact_hard).**
- Zakres: control/cbf.py:309–345, wywołanie w 158–160.
- Wiersze sąsiadów dzielą jeden slack t_s. t_s to minimum, po u w pudełku, maksymalnego naruszenia (faza 1,
  control/cbf.py:326). Ściany i brzegi mają osobny slack t_h (faza 0, control/cbf.py:323).
- Następnie u jest rzutem u_nom na zbiór rozluźniony o t_s i t_h (control/cbf.py:329–341).
- Wagi rho i slack_lin występują tylko w macierzach BoxOSQP (control/cbf.py:121–133). exact_hard ich nie używa:
  slack jest leksykograficzny, a nie ważony.
- Limity u:
  - |u_x|, |u_y| ≤ u_box = 5.657 (control/cbf.py:150, 316–317), czyli V_MAX/√2 przy V_MAX = 8 m/s
    (control/params_legacy.py:71);
  - u_nom jest wcześniej obcinane do ‖·‖ ≤ V_MAX (env/rollout.py:60–61).
- Logowany slack_max to t_s agenta (control/cbf.py:173, 344).

**e) σ_ij a σ_ji: mogą się różnić, a para nie egzekwuje wspólnego marginesu.**
- σ_ij liczy się z okna, które agent i ma o j (compensation/perception.py:245–262). Wchodzą do niego:
  - zaszumiony pomiar range/bearing z perspektywy i;
  - wiek broadcastu j;
  - przemieszczenie własne agenta i (compensation/perception.py:87–113).
- d_safe_pair[i, j] stosuje wyłącznie agent i (control/cbf.py:155–156, 84). Nie ma symetryzacji ani max(σ_ij, σ_ji).
- Podział odpowiedzialności a_ij + a_ji = 1 (control/cbf.py:10–12, 91–95) zakłada wspólne h. Przy różnych d_safe_ij,
  d_safe_ji i różnych x̂ to założenie nie zachodzi.

**f) Cechy głowicy σ (env/rollout.py:161–172): 26 wymiarów.**
- X (20): 4 bloki m = 0..3, każdy (range, bearing, age, ego_disp_x, ego_disp_y) (compensation/perception.py:83–116).
- F (2): wyjście predyktora v3, czyli względna pozycja (compensation/perception.py:262).
- 4 cechy pochodne: |F|, range przeterminowanej obserwacji, |F − stale| i T_frame(N) = N·0.005 s.
- Normalizacja mu/sd z sigma_weights.npz, potem MLP 2×128 ReLU.
- Predyktor v3 to MLP 2×64 na surowym X, bez normalizacji i bez obcięcia wyjścia (compensation/perception.py:71–76).

## 2. ROZKŁAD MARGINESU

Kwantyle d_safe pochodzą z histogramu po ważnych wierszach CBF, z dokładnością koszyka 0.01 m. „≥10.30” oznacza
ucięty górny koszyk, a „max” to najwyższy niepusty koszyk.

σ_c = d_safe − 0.30 tylko powyżej podłogi 0.70. Pod podłogą σ_c ≤ 0.40 i dokładna wartość to BRAK W LOGACH. Dokładne
max σ_c i d_safe per wiersz: BRAK W LOGACH.

Kolumna „mean σ_c” jest dokładna (sig_sum/n). „Max śr./krok” to maksimum średniej d_safe w jednym kroku.

Offline G-cal TEST (binned α = 0.05, pary < 2 m). Zapisane są tylko średnia i p90 σ_c:

| N | mean σ_c | p90 σ_c | odsetek binding |
|---|---|---|---|
| 100 | 0.243 | 0.317 | 0.032 |
| 150 | 0.291 | 0.403 | 0.102 |
| 200 | 0.405 | 0.586 | 0.408 |

Pozostałe kwantyle offline: BRAK W ZAPISANYCH WYNIKACH.

**N=100**

| okno | d_safe p50 / p90 / p95 / p99 / p99.9 / max | mean σ_c | binding | wiersze d_safe > 5 m | max śr./krok |
|---|---|---|---|---|---|
| całość | 0.70 / 0.70 / 0.71 / 0.97 / 1.93 / ≥10.30 | — | 0.054 | 8.9e-5 | — |
| W1 | 0.70 / 0.73 / 0.85 / 1.31 / 2.51 / ≥10.30 | 0.349 | 0.121 | 1.4e-4 | 292 |
| W2 | 0.70 / 0.70 / 0.72 / 1.01 / 2.53 / ≥10.30 | 0.756 | 0.058 | 2.8e-4 | 1724 |
| W3 | 0.70 / 0.70 / 0.70 / 0.83 / 1.39 / 9.69 | 0.243 | 0.038 | 1.8e-5 | 0.73 |
| W4 | 0.70 / 0.70 / 0.70 / 0.80 / 1.15 / 6.84 | 0.239 | 0.033 | 8.6e-7 | 0.72 |
| W5 | 0.70 / 0.70 / 0.70 / 0.80 / 1.14 / ≥10.30 | 0.239 | 0.032 | 1.2e-5 | 1.53 |

σ_c dla całości: p95 0.41, p99 0.67, p99.9 1.63.

**N=150**

| okno | d_safe p50 / p90 / p95 / p99 / p99.9 / max | mean σ_c | binding | wiersze d_safe > 5 m | max śr./krok |
|---|---|---|---|---|---|
| całość | 0.70 / 0.74 / 0.87 / 1.59 / 5.09 / ≥10.30 | — | 0.130 | 1.0e-3 | — |
| W1 | 0.70 / 0.79 / 0.94 / 1.67 / 4.31 / ≥10.30 | 0.519 | 0.174 | 7.9e-4 | 495 |
| W2 | 0.70 / 0.79 / 0.97 / 1.85 / 7.68 / ≥10.30 | 8.00 | 0.168 | 1.5e-3 | 1.31e4 |
| W3 | 0.70 / 0.77 / 0.95 / 2.00 / ≥10.30 / ≥10.30 | 26.6 | 0.150 | 2.0e-3 | 3.48e4 |
| W4 | 0.70 / 0.70 / 0.78 / 1.19 / 3.55 / ≥10.30 | 36.1 | 0.098 | 6.4e-4 | 4.79e4 |
| W5 | 0.70 / 0.70 / 0.75 / 0.97 / 1.98 / ≥10.30 | 0.605 | 0.083 | 1.3e-4 | 675 |

σ_c dla całości: p90 0.44, p95 0.57, p99 1.29, p99.9 4.79.

**N=200**

| okno | d_safe p50 / p90 / p95 / p99 / p99.9 / max | mean σ_c | binding | wiersze d_safe > 5 m | max śr./krok |
|---|---|---|---|---|---|
| całość | 1.23 / 4.11 / 6.97 / ≥10.30 / ≥10.30 / ≥10.30 | — | 0.861 | 0.076 | — |
| W1 | 0.82 / 1.94 / 2.78 / 7.01 / ≥10.30 / ≥10.30 | 1.95 | 0.720 | 0.017 | 2734 |
| W2 | 1.17 / 3.61 / 5.84 / ≥10.30 / ≥10.30 / ≥10.30 | 104.3 | 0.878 | 0.062 | 2.77e5 |
| W3 | 1.41 / 4.79 / 8.34 / ≥10.30 / ≥10.30 / ≥10.30 | 54.8 | 0.908 | 0.094 | 8.98e4 |
| W4 | 1.50 / 5.23 / 9.38 / ≥10.30 / ≥10.30 / ≥10.30 | 85.5 | 0.917 | 0.106 | 8.53e4 |
| W5 | 1.48 / 5.07 / 9.14 / ≥10.30 / ≥10.30 / ≥10.30 | 74.7 | 0.916 | 0.102 | 1.27e5 |

σ_c dla całości: p50 0.93, p90 3.81, p95 6.67, p99 ≥ 10.00. Odsetek wierszy w uciętym koszyku ≥ 10.30 m wynosi 0.032.

## 3. BŁĄD PREDYKCJI

Kwantyle ‖x̂ − x‖ per wiersz, dla wszystkich wierszy i dla < 2 m: **BRAK W LOGACH.**

Zalogowane i podane niżej jest RMS = √MSE ‖x̂ − x‖:
- po wszystkich uporządkowanych parach;
- po parach z prawdziwą odległością < 1 m, ważone parami.

Uwaga: < 1 m, a nie < 2 m.

Referencja TEST (pary < 2 m), średnia / p99 błędu: N100 0.130 / 0.363, N150 0.151 / 0.460, N200 0.208 / 0.675.

Format komórki: RMS wszystkie pary / RMS < 1 m.

| N | ramię | W1 | W2 | W3 | W4 | W5 |
|---|---|---|---|---|---|---|
| 100 | σ | 0.536 / 0.174 | 0.533 / 0.153 | 0.418 / 0.146 | 0.277 / 0.145 | 0.259 / 0.145 |
| 100 | fx | 0.516 / 0.182 | 0.367 / 0.145 | 0.254 / 0.143 | 0.255 / 0.143 | 0.255 / 0.143 |
| 150 | σ | 0.608 / 0.201 | 0.718 / 0.195 | 0.663 / 0.181 | 0.498 / 0.167 | 0.387 / 0.162 |
| 150 | fx | 0.643 / 0.216 | 0.541 / 0.186 | 0.377 / 0.162 | 0.352 / 0.160 | 0.358 / 0.160 |
| 200 | σ | 1.232 / 0.588 | 2.098 / 0.800 | 2.222 / 0.989 | 2.081 / 1.075 | 1.862 / 1.065 |
| 200 | fx | 0.794 / 0.247 | 0.806 / 0.249 | 0.622 / 0.220 | 0.513 / 0.196 | 0.511 / 0.195 |

Pokrycie σ, czyli P(‖x̂ − x‖ ≤ σ_c) w pętli zamkniętej. Format komórki: wszystkie / < 2 m / binding.

| N | W1 | W2 | W3 | W4 | W5 |
|---|---|---|---|---|---|
| 100 | .900 / .912 / .849 | .938 / .942 / .915 | .950 / .953 / .969 | .954 / .954 / .990 | .954 / .954 / .992 |
| 150 | .904 / .912 / .881 | .917 / .926 / .878 | .931 / .941 / .885 | .949 / .952 / .949 | .955 / .956 / .975 |
| 200 | .700 / .745 / .690 | .621 / .683 / .641 | .616 / .679 / .639 | .616 / .677 / .641 | .617 / .679 / .642 |

## 4. PRZESUNIĘCIE CECH

- Cechy głowicy σ i wejścia predyktora w rolloutach: **BRAK W LOGACH**, nie są logowane.
- Zakres p1–p99 na danych treningowych: **BRAK W ZAPISANYCH WYNIKACH.** sigma_weights.npz przechowuje tylko mu i sd 26
  cech. Nie liczono ich ponownie.
- Odsetek wierszy poza zakresem w oknach: **BRAK W LOGACH.**
- Jedyny fakt wynikający z kodu, a nie z logów: cecha T_frame(N) wynosi 0.50 / 0.75 / 1.00 s dla N = 100 / 150 / 200.
  Trening był pooled po N = 25–200, więc 1.00 s to górna wartość treningowa (mu 0.525, sd 0.320).

## 5. WYKONALNOŚĆ I STEROWANIE

Odsetek wierszy z d_safe > prawdziwej odległości: **BRAK W LOGACH.**

Odsetek wierszy z d_safe > d̂ też nie jest zalogowany. Dokładne dolne ograniczenie to odsetek ważnych wierszy z d_safe
> 5 m (punkt 1c). Wartości są w tabelach punktu 2: N200 od 0.017 do 0.106 w oknach, N150 ≤ 0.002, N100 ≤ 3e-4.

Wiersze tabeli poniżej:
- slack śr. — średnie t_s po agentach i krokach;
- akt. — odsetek kroków agenta ze slackiem > 1e-4;
- max — największy slack w oknie;
- ‖u‖ i ‖u_nom‖ — średnie normy;
- pudło — odsetek kroków agenta z |u_x| lub |u_y| na u_box;
- ‖u‖ ≥ 8 — u w narożniku pudełka;
- kolizje — suma par-kroków po 10 seedach.

Odsetek kroków, w których slack ma aktywny choć jeden agent, wynosi 0.98–1.00 w każdym oknie obu ramion.

Prędkość faktyczna ‖Δpos‖/dt jest równa ‖u‖ do 4 cyfr, bo ν = 0, w każdej komórce.

row_violation max: σ 6.1e-5, fx 1.1e-5. QP spełnia rozluźnione wiersze w tolerancji.

**N=100**

| wiersz | W1 σ / fx | W2 σ / fx | W3 σ / fx | W4 σ / fx | W5 σ / fx |
|---|---|---|---|---|---|
| slack śr. | 0.702 / 0.106 | 0.927 / 0.116 | 0.152 / 0.095 | 0.129 / 0.084 | 0.145 / 0.074 |
| slack akt. | 0.229 / 0.161 | 0.262 / 0.229 | 0.204 / 0.205 | 0.192 / 0.189 | 0.170 / 0.174 |
| slack max | 1000 / 20.9 | 1000 / 10.5 | 487 / 6.8 | 763 / 5.1 | 994 / 4.4 |
| ‖u‖ | 1.557 / 1.493 | 1.199 / 0.996 | 1.011 / 0.910 | 0.977 / 0.916 | 0.932 / 0.896 |
| ‖u_nom‖ | 6.30 / 5.40 | 3.12 / 2.14 | 2.02 / 1.55 | 1.64 / 1.65 | 1.75 / 1.76 |
| pudło | 0.060 / 0.022 | 0.035 / 0.015 | 0.017 / 0.002 | 0.009 / 0.001 | 0.005 / 0.001 |
| ‖u‖ ≥ 8 | 0.0039 / 1e-5 | 0.0032 / 0 | 0.0017 / 2e-6 | 0.0017 / 4e-7 | 0.0016 / 4e-7 |
| kolizje | 10671 / 331 | 10911 / 0 | 10 / 0 | 0 / 0 | 0 / 0 |

**N=150**

| wiersz | W1 σ / fx | W2 σ / fx | W3 σ / fx | W4 σ / fx | W5 σ / fx |
|---|---|---|---|---|---|
| slack śr. | 1.20 / 0.083 | 2.89 / 0.098 | 4.57 / 0.127 | 1.55 / 0.123 | 0.487 / 0.112 |
| slack akt. | 0.223 / 0.150 | 0.227 / 0.189 | 0.241 / 0.254 | 0.245 / 0.251 | 0.243 / 0.233 |
| slack max | 1000 / 117 | 1003 / 98.6 | 1004 / 8.8 | 1004 / 6.8 | 1000 / 6.3 |
| ‖u‖ | 1.427 / 1.397 | 1.382 / 1.154 | 1.324 / 1.002 | 1.117 / 0.979 | 1.041 / 0.964 |
| ‖u_nom‖ | 7.13 / 6.49 | 5.17 / 3.62 | 3.59 / 2.50 | 2.75 / 2.31 | 2.20 / 2.12 |
| pudło | 0.042 / 0.020 | 0.044 / 0.013 | 0.040 / 0.005 | 0.018 / 0.001 | 0.010 / 0.001 |
| ‖u‖ ≥ 8 | 0.0077 / 7e-5 | 0.0093 / 5e-5 | 0.0103 / 2e-6 | 0.0039 / 2e-6 | 0.0020 / 1e-6 |
| kolizje | 5030 / 5508 | 5529 / 3025 | 5200 / 14 | 1846 / 0 | 357 / 34 |

**N=200**

| wiersz | W1 σ / fx | W2 σ / fx | W3 σ / fx | W4 σ / fx | W5 σ / fx |
|---|---|---|---|---|---|
| slack śr. | 37.6 / 0.184 | 100.2 / 0.148 | 148.8 / 0.130 | 167.2 / 0.136 | 164.0 / 0.131 |
| slack akt. | 0.783 / 0.264 | 0.844 / 0.213 | 0.847 / 0.229 | 0.850 / 0.258 | 0.846 / 0.251 |
| slack max | 1005 / 537 | 1005 / 374 | 1005 / 396 | 1005 / 23.0 | 1005 / 30.1 |
| ‖u‖ | 3.93 / 1.353 | 5.20 / 1.297 | 5.51 / 1.171 | 5.66 / 1.076 | 5.68 / 1.052 |
| ‖u_nom‖ | 7.36 / 7.03 | 6.61 / 5.29 | 6.03 / 3.96 | 5.46 / 3.41 | 4.92 / 3.10 |
| pudło | 0.343 / 0.015 | 0.551 / 0.011 | 0.622 / 0.006 | 0.652 / 0.002 | 0.651 / 0.001 |
| ‖u‖ ≥ 8 | 0.155 / 1.4e-4 | 0.272 / 1.7e-4 | 0.324 / 7.9e-5 | 0.351 / 4.9e-6 | 0.357 / 4.8e-6 |
| kolizje | 867049 / 22806 | 734470 / 24447 | 455160 / 8074 | 371491 / 370 | 364224 / 474 |

Sumy kolizji (par-kroki, 10 seedów), σ / fx:
- N100: 21592 / 331;
- N150: 17962 / 8581;
- N200: 2792394 / 56171.

## 6. SKOKI MARGINESU

- |Δd_safe| na krok dla tej samej pary: **BRAK W LOGACH.**
- |σ_ij − σ_ji| dla par widzących się nawzajem: **BRAK W LOGACH.**

Zalogowany jest tylko agregat: średnia d_safe w kroku (d_safe_sum/d_safe_n). Rozkład |Δ| tej średniej między kolejnymi
krokami, po 10 seedach:

| N | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|
| 100 | 0.0010 | 0.0037 | 0.0058 | 0.035 | 9.8 | 1703 |
| 150 | 0.0024 | 0.035 | 0.40 | 26.3 | 1949 | 3.48e4 |
| 200 | 0.44 | 8.5 | 22.1 | 216 | 5411 | 2.77e5 |

**Zależność od fazy ramki TDMA.** Ramka ma F = 50 / 75 / 100 kroków. Średnie |Δ| liczono według fazy k mod F. Stosunek
max do średniej po fazach:
- N100: 7.1 (faza 27);
- N150: 2.8 (faza 33);
- N200: 5.6 (faza 1).

Każdy agent ma własną fazę slotu, więc per-agentowej zależności od wieku broadcastu nie da się z logów wyznaczyć:
**BRAK W LOGACH.**

**Korelacja Δ(średnia d_safe w kroku) z Δ(średnie ‖u‖)** w kroku k + L, dla L = 0, 1, 2, 5, 10:
- N100: 0.029, −0.012, −0.001, 0.004, −0.001;
- N150: 0.004, −0.005, 0.002, −0.004, 0.000;
- N200: 0.000, −0.001, 0.003, −0.002, 0.005.

## 7. KOLEJNOŚĆ ZDARZEŃ (kroki od 0, dt = 0.01 s)

Kryteria:
- p99 ‖x̂ − x‖ > p99 TEST: **BRAK W LOGACH.** Jako wskaźnik zastępczy, a nie p99, podano pierwszy krok, w którym RMS
  ‖x̂ − x‖ po parach < 1 m przekracza p99 TEST.
- p99 σ_c > 1 m: wyznaczone dokładnie, jako p99 d_safe w kroku > 1.30 m.
- ‖u‖: pierwszy krok, w którym średnie ‖u‖ σ przekracza 1.5 × średnie ‖u‖ fx w tym samym seedzie i kroku.
- Kolizje: pierwsza kolizja σ, a w nawiasie pierwsza kolizja fx.

**N=100**

| seed | p99 σ_c > 1 m | ‖u‖ > 1.5× fx | 1. kolizja σ (fx) | RMS zastępczy σ / fx |
|---|---|---|---|---|
| 0 | 39 | 2151 | 92 (205) | — / 35 |
| 1 | 31 | 1959 | 40 (391) | — / — |
| 2 | 40 | 2171 | 151 (1118) | — / — |
| 3 | 48 | 1990 | 143 (42) | — / — |
| 4 | 34 | 1857 | 150 (83) | 33 / 35 |
| 5 | 35 | 1857 | 81 (356) | — / 34 |
| 6 | 45 | 1983 | 175 (888) | — / — |
| 7 | 42 | 1925 | 44 (39) | — / — |
| 8 | 45 | 1964 | 97 (37) | — / — |
| 9 | 32 | 1975 | 32 (31) | 36 / 27 |

**N=150**

| seed | p99 σ_c > 1 m | ‖u‖ > 1.5× fx | 1. kolizja σ (fx) |
|---|---|---|---|
| 0 | 58 | 4196 | 213 (163) |
| 1 | 59 | 2720 | 137 (187) |
| 2 | 44 | 4223 | 98 (88) |
| 3 | 61 | 4209 | 419 (153) |
| 4 | 55 | 3933 | 224 (105) |
| 5 | 48 | 4356 | 132 (199) |
| 6 | 51 | 4061 | 90 (104) |
| 7 | 45 | 4196 | 55 (363) |
| 8 | 55 | 3716 | 191 (135) |
| 9 | 61 | 3758 | 153 (278) |

RMS zastępczy przy N=150 nie przekracza progu w żadnym seedzie, w obu ramionach.

**N=200**

| seed | p99 σ_c > 1 m | ‖u‖ > 1.5× fx | 1. kolizja σ (fx) | RMS zastępczy σ |
|---|---|---|---|---|
| 0 | 33 | 274 | 70 (73) | 894 |
| 1 | 32 | 250 | 58 (142) | 837 |
| 2 | 32 | 399 | 116 (193) | 1074 |
| 3 | 44 | 291 | 32 (141) | 783 |
| 4 | 28 | 282 | 37 (137) | 925 |
| 5 | 32 | 563 | 176 (132) | 1418 |
| 6 | 41 | 247 | 45 (73) | 861 |
| 7 | 30 | 236 | 37 (236) | 1075 |
| 8 | 41 | 292 | 95 (140) | 1030 |
| 9 | 28 | 429 | 20 (21) | 985 |

RMS zastępczy fx przy N=200 nie przekracza progu w żadnym seedzie.

**Co zwykle jest pierwsze** (bez kryterium p99 błędu, którego nie ma w logach):
- p99 σ_c > 1 m jest pierwsze w 9/10 seedów przy N=100 (seed 9: remis z kolizją w kroku 32), 10/10 przy N=150 i 8/10
  przy N=200 (w seedach 3 i 9 pierwsza jest kolizja).
- Przekroczenie p99 σ_c > 1 m następuje w krokach 28–61 przy każdym N, także przy N=100, gdzie ramię działa
  najlepiej.
- ‖u‖ > 1.5× fx występuje zawsze później niż pierwsza kolizja: N100 1857–2171, N150 2720–4356, N200 236–563.
- Zastępczy RMS błędu przy N=200 przekracza próg dopiero w krokach 783–1418, czyli po σ, kolizji i ‖u‖.

**Korelacje z opóźnieniem** (Pearson na szeregach per krok, lag ±300 kroków co 5, per seed). Szeregi: RMS błędu
(wszystkie pary), średnie σ_c wierszy, średnie ‖u‖ i liczba kolizji. Podano średnie r przy lagu 0:

| N | błąd→σ | σ→‖u‖ | ‖u‖→kolizje | błąd→kolizje | σ→kolizje |
|---|---|---|---|---|---|
| 100 | 0.18 | 0.23 | 0.39 | 0.43 | 0.12 |
| 150 | 0.06 | 0.04 | 0.25 | 0.22 | 0.05 |
| 200 | 0.03 | 0.03 | −0.06 | −0.21 | −0.02 |

Lagi maksymalnej |r| leżą przeważnie na brzegu przeszukiwania (±280 do ±300). Przy N=200 dla ‖u‖→kolizje i
błąd→kolizje jest tak we wszystkich 10 seedach, z r ujemnym. Nie wyznaczają więc stabilnego wyprzedzenia. Szczegóły per
seed są w diag.json („lagcorr”).

## 8. NIEDETERMINIZM learned (audyt kodu i konfiguracji; dane pary g2)

**Konfiguracja**
- logs/env.sh na podzie, odczytany przed jego zatrzymaniem: `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Łańcuch amend31 ani add4 nie ustawia `XLA_FLAGS`. Nie ma `--xla_gpu_autotune_level=0` ani
  `--xla_gpu_deterministic_ops`, więc autotuning GEMM działa w trybie domyślnym.
- Autotune 0 występuje w repo tylko w logs/amend19_pod/amend19_presolve_sanity.sh:29.
- Precyzja matmul to highest (experiments/ladder.py:82).
- Wersja jax/jaxlib na tym podzie: **BRAK.** SSH został odrzucony po przejściu poda w EXITED.

**RNG**
- Klucze jax.random są dzielone deterministycznie (env/rollout.py:49, 91, 142).
- Szum procesu jest mnożony przez ν = 0.
- Ten sam klucz daje te same losowania, więc RNG nie jest źródłem rozjazdu.

**Kolejność operacji, możliwe źródła**
- Wybór algorytmu GEMM przez autotuner w każdym procesie: predyktor v3 w perceive i learned_window_and_pred oraz
  głowica σ.
- Redukcje zmiennoprzecinkowe na GPU.
- Scatter-add histogramu działa na int32, więc jest deterministyczny.
- Wrapper add4 przy braku klucza nie dotyka obiektu scenariusza (experiments/amend31_add4_run.py:21–33).

**Para g2** (baza N150 sigma_bin seed 0 a przebieg add4 bez perturbacji), w porządku pierwszych różnic:
- Krok 0, pierwsza różnica w logach:
  - u agenta 44, składowa x: −0.15789652 a −0.15789708 (Δ 5.7e-7);
  - slack_max i intervention agenta 44;
  - xhat_err_all: 0.11691106 a 0.11691105.
  - Oznacza to, że x̂, czyli wyjście predyktora, różni się już w kroku 0.
- Krok 1: row_violation. Krok 2: xhat_err_near.
- Kroki 20 i 25: pos, u_nom, m_hat, rho_true, potem m_star_at.
- Krok 36: d_safe_sum i sig_sum. Krok 254: d_safe_hist. Krok 1290: n_coll_pairs.
- pos różni się dopiero od kroku 20. Różnica dt·Δu ≈ 6e-9 jest mniejsza od rozdzielczości f32 pozycji (~4.8e-7 przy
  4 m), dopóki nie narośnie.

# HIPOTEZY

**H1 — brak limitu σ powoduje rozbieganie: NIEROZSTRZYGNIĘTA.**
- Brak limitu jest POTWIERDZONY w kodzie (1a) i w danych (2):
  - średnia d_safe w kroku do 2.77e5 m przy N200, 4.79e4 przy N150 i 1724 przy N100;
  - przy N200 3.2 % wierszy w koszyku ≥ 10.30 m;
  - przy N200 d_safe p50 rośnie w oknach 0.82 → 1.50, a binding 0.72 → 0.92.
- Przyczynowości nie da się sprawdzić z logów bez interwencji.
- Prawa strona wiersza jest i tak obcięta do 1000 (1a, 1c). Każdy d_safe > d̂ już czyni wiersz niespełnialnym, a
  wielkość σ ponad to nie zmienia wiersza.

**H2 — niespełnialne wiersze i slack powodują skoki u: NIEROZSTRZYGNIĘTA.**
- Przy N200 współwystępują (5):
  - ≥ 1.7–10.6 % wierszy z d_safe > d̂;
  - slack śr. 37.6–167, a fx 0.13–0.18;
  - slack nasycony na ~1000;
  - u na pudle w 34–65 % kroków agenta, a fx ≤ 1.5 %;
  - ‖u‖ 3.9–5.7, a fx 1.05–1.35.
- Związku per wiersz i per agent nie policzono. Pod przeszedł w EXITED, a lokalna kopia logów się nie zmieściła.
- Agregatowa korelacja Δd_safe z Δ‖u‖ wynosi |r| ≤ 0.03 (6). „Skoki” nie zostały pokazane.

**H3 — brak członu czasowego przy zmiennym marginesie destabilizuje CBF: NIEROZSTRZYGNIĘTA.**
- Brak członu jest POTWIERDZONY w kodzie (1b).
- Agregatowe skoki średniej d_safe są duże: p99 216 m na krok przy N200 i 26 przy N150 (6).
- Skoki per para: BRAK W LOGACH. Korelacja ze zmianą u na poziomie agregatu ≈ 0.

**H4 — asymetria σ_ij i σ_ji psuje koordynację: NIEROZSTRZYGNIĘTA.**
- Kod dopuszcza asymetrię i nie wymusza wspólnego marginesu (1e).
- |σ_ij − σ_ji|: BRAK W LOGACH (6).

**H5 — najpierw rośnie błąd predyktora poza zakresem treningowym, a σ to tylko odzwierciedla: NIEROZSTRZYGNIĘTA.**
- p99 błędu per wiersz i cechy: BRAK W LOGACH (3, 4).
- Wskaźnik zastępczy przemawia przeciw kolejności „najpierw błąd”:
  - przy N200 RMS < 1 m przekracza p99 TEST dopiero w krokach 783–1418, po p99 σ_c > 1 m (28–44) i po pierwszej
    kolizji, w 10/10 seedach (7);
  - w W1 RMS σ wynosi 0.59, a fx 0.25 (3), i rośnie potem do 1.07, podczas gdy fx spada do 0.20.
- To jednak RMS, a nie p99 i nie cechy, więc hipoteza pozostaje nierozstrzygnięta.

**H6 — szkoda przy N=100 mimo małej liczby wierszy wiążących, innym mechanizmem niż przy N=200.**
- **Część 1 (szkoda mimo małej liczby wierszy wiążących): POTWIERDZONA.**
  - Binding przy N100: 5.4 % wierszy łącznie, 12.1 % w W1 (2).
  - Kolizje: σ 21592 a fx 331, prawie wyłącznie w W1–W2 (5).
- **Część 2 (inny mechanizm): NIEROZSTRZYGNIĘTA.** Obraz zjawiska się różni:
  - N100: ‖u‖ σ / fx = 1.04 w W1, u na pudle 6 % a 2 %, slack śr. 0.70 a 0.11, pokrycie < 2 m 0.91–0.95;
  - N200: ‖u‖ 3–5×, pudło 34–65 %, slack 37–167, pokrycie 0.68–0.75 (3, 5).
  - Mechanizmu przy N=100 nie zidentyfikowano. Brak danych per wiersz.

# CZEGO NIE ZROBIONO I DLACZEGO

- **Kwantyle per wiersz** σ_c, d_safe i ‖x̂ − x‖, cechy, skoki per para, |σ_ij − σ_ji|, wiersze z d_safe > prawdziwej
  odległości: **BRAK W LOGACH** (log_xhat = false). Nie odtwarzano ich ponownym liczeniem.
- **Zakresy p1–p99 cech treningowych i kwantyle G-cal TEST** poza zapisanymi (mean i p99 e; mean i p90 σ_c): nie są
  zapisane. Nie liczono ich ponownie.
- **Analiza per agent związku slack → u** (np. udział pudła przy slack > 1 i ≤ 1e-4), choć slack_max i u per agent są w
  logach: nie wykonano.
  - Lokalna kopia npz (4.2 GB) nie zmieściła się na dysku (1.2 GB wolne). Częściową kopię usunąłem.
  - Pod przeszedł w EXITED po zakończeniu głównej analizy, więc logi są niedostępne.
- **Wersja jax/jaxlib na podzie:** niedostępna z tego samego powodu.
- **Skrypt analizy** (numpy) uruchomiono w /tmp poda, bez zapisu na wolumenie. Wynik jest w
  results/amend31_add4/diag.json. Skryptu nie commitowano.

# POD

Stan z get-pod: **EXITED**.
- sbk37l6nvrg36q (musc-a100), A100-SXM4-80GB, US-MD-1, 1.79 USD/h, startedAt 2026-11-02T14:12:33Z.
- Przy odczycie przez SSH około 21:45 UTC działał bezczynnie: GPU 0 %, 0 MiB, brak sesji tmux. Działały tylko sshd,
  nginx i `sleep infinity`, poza analizą numpy z tego zadania.
- Później SSH zwrócił „Connection refused”. Nie zatrzymywałem poda; przyczyna przejścia w EXITED jest nieznana.

Koszt (list-pod-billing, koszyk 2026-11-02): **12.22 USD** (GPU 11.94 + dysk 0.27).

Pod zostawiono w stanie EXITED. Nie uruchamiano go ani nie usuwano; wolumen jest nietknięty.
