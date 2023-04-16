class: center, middle, inverse
<style>	.remark-code, code { padding: 5px; font-family: monospace; font-size: 15px;} </style>
<style> img { max-height: 520px; } </style>

# Algorytmy w inżynierii danych

## Wykład 07 - Macierze i wartości własne

## Bartosz Chaber

e-mail: bartosz.chaber@pw.edu.pl
2023L

.img-nerw-header[![nerw](https://www.nerw.pw.edu.pl/var/nerw/storage/images/pasek-logo/4045-3-pol-PL/Pasek-logo.png)]

---
### Macierzowe układy równań
Jest wiele ciekawych problemów, które można rozwiązać znajdując
taki wektor $\mathbf{x}$, który spełnia macierzowy układ równań:

$$\mathbf{A} \mathbf{x} = \mathbf{b}$$,
gdzie znamy macierz $\mathbf{A}$, oraz pewien wektor $\mathbf{b}$.

---
Renderowanie sceny 3D metodą Radiosity
---
Symulowanie rozchodzenia się fal
---

---
### Macierz, jako przekształcenie

---
### Wartości i wektory własne
Wektory własne są charakterystycznymi wektorami dla konkretnej, kwadratowej macierzy $\mathbf{A}$, spełniającymi tzw. równanie charakterystyczne:

$$\mathbf{A} \mathbf{v} = \lambda \mathbf{v}$$

Wektorów i wartości własnych jest tyle, co liczba kolumn macierzy $\mathbf{A}$.
Niektóre z wartości własnych mogą się powtarzać (są to wielokrotne wartości własne).

---
### Macierz diagonalna
(i macierz jednostkowa)

---
### Macierz sprzężona

Transpozycja
Sprzężenie zespolone
Sprzężenie hermitowskie
```julia
> A = [1 2 3; 4 5 6; 7 8 9]
3×3 Matrix{Int64}:
 1  2  3
 4  5  6
 7  8  9

> A' # transpozycja (zamiana wierszy z kolumnami)
3×3 adjoint(::Matrix{Int64}) with eltype Int64:
 1  4  7
 2  5  8
 3  6  9

> A = [1 1im; 0 1im]
2×2 Matrix{Complex{Int64}}:
 1+0im  0+1im
 0+0im  0+1im

> conj(A) # sprzężenie zespolone
2×2 Matrix{Complex{Int64}}:
 1+0im  0-1im
 0+0im  0-1im

> A' # sprzężenie hermitowskie = sprzężenie zespolone + transpozycja
2×2 adjoint(::Matrix{Complex{Int64}}) with eltype Complex{Int64}:
 1+0im  0+0im
 0-1im  0-1im
```

---
### Macierz Hermitowska
Jeżeli macierz $\mathbf{A}$ jest równa swojej macierzy sprzężonej
$\mathbf{A}^H$ to każda wartość własna jest liczbą rzeczywistą.

---
### Macierz symetryczna
Ponieważ macierz symetryczna z elementami rzeczywistymi jest macierzą Hermitowską to jej wartości własne są liczbami rzeczywistymi.

---
### Macierz dodatnio określona
Jeżeli macierz symetryczna $\mathbf{A}$ jest dodatnio/ujemnie określona wszystkie jej wartości własne są dodatnie/ujemne.

---
### Macierz osobliwa
Jeżeli macierz $\mathbf{A}$ jest osobliwa, to co najmniej jedna wartość własna jest zerowa.

---
### Macierz odwrotna
Jeżeli macierz jest nieosobliwa (czyli jest odwracalna), to wartości własne macierzy odwrotnej są odwrotnościami macierzy oryginalnej.

---
### Macierz rzadka

---
## Literatura
---
class: center, middle, inverse
# Dziękuję za uwagę
