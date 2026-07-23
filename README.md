# Motion Classification Project

Projekat se bavi obradom motion-capture podataka i klasifikacijom pokreta ruku.

## Ciljevi projekta

1. Rekurzivno pronalaženje i učitavanje CSV fajlova.
2. Prepoznavanje vrste pokreta na osnovu naziva fajla i putanje.
3. Izdvajanje karakteristika:
   - rastojanje između šaka,
   - položaj šaka u odnosu na trup,
   - brzina,
   - ubrzanje.
4. Vremenska normalizacija pokreta.
5. Klasifikacija pokreta pomoću SVM algoritma.
6. Primena funkcionalne PCA ako obična vremenska normalizacija nije dovoljna.

## Pokreti

Početna klasifikacija obuhvata dva pokreta:

- širenje ruku u stranu,
- pružanje ruku napred.

## Struktura projekta

- `data/raw` – originalni CSV podaci
- `data/interim` – privremeno obrađeni podaci
- `data/processed` – podaci spremni za klasifikaciju
- `src` – Python kod
- `notebooks` – eksperimenti i analiza
- `models` – sačuvani modeli
- `results` – rezultati i grafikoni
- `tests` – automatski testovi

## Instalacija

```bash
python -m venv .venv
pip install -r requirements.txt