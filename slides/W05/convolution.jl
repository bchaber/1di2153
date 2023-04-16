### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 9ae9879e-a377-11eb-084e-09139cd358a6
begin using TestImages, Colors, GR end

# ╔═╡ b3e17c32-8850-43ac-ae58-565bb553f4b8
begin
	using FFTW, SparseArrays, BenchmarkTools, StaticArrays
	md"""
	#### Szybkie algorytmy konwolucji
	"""
end

# ╔═╡ 33289b77-d488-4e70-9c55-29a5344ed08a
md"""#### Operacja splotu
Dzisiaj przyjrzymy się ciekawemu narzędziu, czyli _konwolucji_.

W większości przypadków mówiąc o splocie (to inna nazwa konwolucji) mamy na myśli, np. nakładanie filtru na obraz lub przetwarzanie dźwięku.

Dlatego rozpoczniemy od przygotowania typowego obrazu kamerzysty.
"""

# ╔═╡ ec947bb1-249f-48a9-88d2-9ecb5370888e
cameraman = testimage("cameraman") .|> gray .|> Float64;

# ╔═╡ 7474bb15-a601-47c4-b161-9b86248bdce3
let
	figure(size=(250, 250))
	imshow(cameraman)
end

# ╔═╡ 646ffce2-8c16-4e5f-88e2-c6ef585cac1e
md"""
Zdefiniujmy teraz naiwną funkcję, która oblicza splot maski `K` z obrazem `I`.
"""

# ╔═╡ 4449113d-1849-4b43-a159-823c41030f88
function conv(I :: Array{Float64,1}, K :: Array{Float64, 1})
	n = length(I) - 1
	J = zeros(n+1)
	for i=2:n
		J[i] = sum(I[i-1:i+1] .* K)
	end

	return J
end

# ╔═╡ 475f9ac7-e2ea-4765-a45e-56b6d949fb62
function conv(I :: Array{Float64, 2}, K :: Array{Float64, 2})
	n, m = size(I) .- 1
	J = zeros(n+1, m+1)
	for i=2:n, j=2:m
		J[i, j] = sum(I[i-1:i+1, j-1:j+1] .* K)
	end
	
	return J
end

# ╔═╡ 81f30430-eef5-427c-8e32-c291a7699967
function conv(I :: Array{Float64, 3}, K :: Array{Float64, 3})
	n, m, _ = size(I) .- 1
	l = size(I, 3)
	J = zeros(n+1, m+1)
	for i=2:n, j=2:m, k=1:l
		J[i, j] = sum(I[i-1:i+1, j-1:j+1, 1:l] .* K)
	end
	
	return J
end

# ╔═╡ 3175d110-fc9b-44cf-9c79-1210be843eb6
md"""
Sprawdźmy jak mogą wyglądać efekty takiej operacji dla maski:

$ K = 
  \begin{bmatrix}
    1/9 & 1/9 & 1/9 \newline
    1/9 & 1/9 & 1/9 \newline
    1/9 & 1/9 & 1/9
\end{bmatrix} $.
"""

# ╔═╡ 30f2260c-a6c2-412c-a855-0d756df4561a
let
	img = cameraman[100:200, 200:300]
	
	figure(size=(502, 250))
	subplot(1, 2, 1); imshow(img)
	subplot(1, 2, 2); imshow(conv(img, [0/9 0/9 0/9; 0/9 0/9 0/9; 10/9 0/9 1/9]))
end

# ╔═╡ 400061db-f956-4408-97d5-87218a08625f
md"""
Możemy też spróbować inne jądro. Jest to tzw. operator Sobela, służący do wykrywania krawędzi.

$ K = 
  \begin{bmatrix}
    -1 & -2 & -1 \newline
     0 &  0 &  0 \newline
    +1 & +2 & +1
\end{bmatrix} $ lub $ K = 
  \begin{bmatrix}
    -1 &  0 & +1 \newline
    -2 &  0 & +2 \newline
    -1 &  0 & +1
\end{bmatrix} $.
"""

# ╔═╡ 101baac9-8bab-47f8-9f30-a987b5eda720
let
	img = cameraman[100:200, 200:300]
	
	figure(size=(755, 250))
	subplot(1, 3, 1); imshow(img)
	subplot(1, 3, 2); imshow(conv(img, [-1 -2 -1; 0 0.  0; +1 +2 +1]))
	subplot(1, 3, 3); imshow(conv(img, [-1  0 +1;-2 0. +2; -1  0 +1]))
end

# ╔═╡ 94b67c4a-5f79-4aca-8fbc-98ad3df573a1
begin
	N = 31
	x = repeat(0:1/N:1, 1, N+1)
	y = rotr90(x)
	r = sqrt.((x .- 0.5).^2 .+ (y .- 0.69).^2)
	u = r .< 0.20
	
	g(r, ct) = exp.(-200(r .- ct).^2)
	md"""#### Równania różniczkowe cząstkowe
	Spróbujmy teraz sami stworzyć obraz. Niech będzie to koło o promieniu 0.2_m_.
	"""
end

# ╔═╡ 8d186053-3329-42ee-8b3b-df3d4a9cbdbf
begin
	glider = [0 0 1
		  	  1 0 1
		  	  0 1 1]
	
	beard = [1 1 0 1 1
		 	 1 0 0 0 1
		 	 0 1 1 1 0]
	
	ls = zeros(N+1, N+1, 50)
	ls[9:11,15:19,1] .= beard
	
	M = [1  1  1
		 1 10. 1
	 	 1  1  1]
	
	gol(n) = (n == 3) || (12 <= n <= 13) ? 1 : 0
	
	for t=2:50
		ls[:, :, t] .= gol.(conv(ls[:, :, t-1], M))
	end
	
	md"""#### Automaty komórkowe
	Zasady automatów komórkowych można potraktować jako operację splotu na
	siatce/planszy. Jako przykład zaimplementujemy Grę w Życie Johna Conwaya.
	"""
end

# ╔═╡ b9388c50-7ea3-4b45-9206-855750840b1b
let
	figure(size=(305, 150))
	subplot(1,2,1); title("Tophat"); imshow(u)
	subplot(1,2,2); title("Gaussian"); imshow(g(r, 0.2))
end

# ╔═╡ ee55200c-1ee1-49ed-83f2-579dfbe97063
begin # diffusion equation
	T  = 150
	Δt = 1e-4
	Δx = 1.0 / N
	α  = .1Δt / Δx^2
	
	us = zeros(N+1, N+1, T)
	us[:, :, 1] = u
	
	D = zeros(3, 3, 1)
	D[:, :, 1] =[0  1  0
				 1 -4. 1
				 0  1  0] * α
	D[2, 2, 1]+= 1.0
	for t=2:T
		us[:, :, t] .= conv(us[:, :, t-1:t-1], D) 
	end
end

# ╔═╡ 10b25830-6bc8-4f25-937a-385330ed95d6
md"""
Niech ten obraz będzie pierwszym na "stosie" obrazów. Każdy kolejny powstanie jako nałożenie na ostatni obraz wyniku jego konwolucji z jądrem:

$ K = 
  \alpha\begin{bmatrix}
    0 & 1 & 0 \newline
    1 &-4 & 1 \newline
    0 & 1 & 0
\end{bmatrix} + \begin{bmatrix}
    0 & 0 & 0 \newline
    0 & 1 & 0 \newline
    0 & 0 & 0
\end{bmatrix}$, gdzie współczynnik $ \alpha \approx $ $(α)
"""

# ╔═╡ 94dad3ac-62fb-4eb2-9859-c5fc8e44e520
md"""
Żeby było przyjemniej, dodajmy sobie suwak, który pozwoli nam odwiedzać wybrany obraz ze stosu.
"""

# ╔═╡ 136147d8-aab2-446d-ba76-72662f992144
import PlutoUI: Slider, NumberField

# ╔═╡ e86d9240-ac1f-422c-a5ee-e5132dd651d7
let
	md"""
	Epoch:
	$(@bind epoch NumberField(1:50))
	"""
end

# ╔═╡ f6d2e60f-88e0-47c9-9126-fd2c4ffd6186
let
	figure(size=(250, 250))
	imshow(ls[:, :, epoch])
end

# ╔═╡ 5073dc2c-7d4f-42bb-9ff8-d83d3e708185
let
	md"""
	$(@bind slice Slider(1:T))
	slice number:
	$(@bind slice NumberField(1:T))
	"""
end

# ╔═╡ fe58460c-4e71-486d-ab4d-b7b3d9e683d6
begin # advection equation
	β = 15Δt / Δx
	
	vs = zeros(N+1, N+1, T)
	vs[:, :, 1] = u
	
	A = zeros(3, 3, 1)
	A[:, :, 1] =[0  1  0
				 1 -2. 0
				 0  0  0] * β
	A[2, 2, 1]+= 1.0
	
	for t=2:T
		vs[:, :, t] .= conv(vs[:, :, t-1:t-1], A) 
	end
end

# ╔═╡ 919fe04a-e442-4b05-8641-74e2015ea58e
md"""
Zmodyfikujmy delikatnie jądro splotu. Otrzymamy wtedy "trochę" inne zachowanie.

$ K = 
  \beta\begin{bmatrix}
    0 & 1 & 0 \newline
    1 &-2 & 0 \newline
    0 & 0 & 0
\end{bmatrix} + \begin{bmatrix}
    0 & 0 & 0 \newline
    0 & 1 & 0 \newline
    0 & 0 & 0
\end{bmatrix} $, gdzie współczynnik $ \beta \approx $ $(β)
"""

# ╔═╡ 68d9573f-8cf0-4b2d-a237-0de3cf7e84c0
begin # wave equation
	c = 100Δt / Δx
	
	ws = zeros(N+1, N+1, T)
	ws[:, :, 1] = g(r, .0 + .0*c)
	ws[:, :, 2] = g(r, .0 + Δt*c)
	
	W = zeros(3, 3, 2)
	W[:, :, 2] =[0  1  0
				 1 -4. 1
				 0  1  0] * c^2
	W[2, 2, 2]+= 2.0
	W[2, 2, 1]-= 1.0
	for t=3:T
		ws[:, :, t] .= conv(ws[:, :, t-2:t-1], W) 
	end
end

# ╔═╡ 6d623f99-9f2e-4568-9348-0f4e1101fcb7
let
	figure(size=(755, 250))
	subplot(1,3,1); title("Waves");     imshow(ws[:, :, slice])
	subplot(1,3,2); title("Advection"); imshow(vs[:, :, slice])
	subplot(1,3,3); title("Diffusion"); imshow(us[:, :, slice])
end

# ╔═╡ 8a0dc096-0c8c-48e8-99e0-b978d359f6a4
md"""
A teraz mój ulubiony przykład! Zwróćmy uwagę, że teraz zaczynamy uzupełniać stos obrazów od trzeciego, ponieważ wymagamy *dwóch* obrazów wstecz.

$ K = 
  c^2\begin{bmatrix}
    0 & 1 & 0 \newline
    1 &-4 & 1 \newline
    0 & 1 & 0
\end{bmatrix} + \begin{bmatrix}
    0 & 0 & 0 \newline
    0 & 2 & 0 \newline
    0 & 0 & 0
\end{bmatrix}$, gdzie współczynnik $ c^2 \approx $ $(c^2)
"""

# ╔═╡ c1d47a9e-ba03-42e2-9fc9-8c6c8e5bfefd
md"""
Poprzednie równania charakteryzowało to, że polegały one na wcześniejszym "stanie", który ulegał zmianie zgodnie z postacią równania.
Inaczej ma miejsce w przypadku równania Poissona, które jest niezmiennicze w czasie (opisuje stan ustalony).

Natomiast, możliwe jest zapisanie procesu iteracyjnego pozwalającego na rozwiązanie równania Poissona. W poniższym przykładzie wykorzystamy iterację Jacobiego.

$ K = 
  \frac{1}{4}\begin{bmatrix}
    0 & 1 & 0 \newline
    1 & 0 & 1 \newline
    0 & 1 & 0
\end{bmatrix} $.
"""

# ╔═╡ 4743da0e-b2a1-4b5f-b2ae-bcd5cde62b80
begin # Poisson equation (Jacobi iteration) 0.996581*
	ps = zeros(N+1, N+1, 1500)
	b  =-8π^2 * sin.(2π.*x).*sin.(2π.*y) * Δx^2
    ps[:, :, 1] = randn(N+1, N+1)
	
	P = [ 0 1 0
		  1 0 1
		  0 1 0.]

	for t=2:1500
		ps[:, :, t] .= conv(ps[:, :, t-1], P) .- b
		ps[:, :, t]./= 4
	end
end

# ╔═╡ ee0a8cc5-5c55-4925-a48b-f89aa1aeb69f
let
	figure(size=(505, 525))
	subplot(2, 2, 1); title("1 iteration"); imshow(ps[:, :, 1])
	subplot(2, 2, 2); title("10 iteration"); imshow(ps[:, :, 10])
	subplot(2, 2, 3); title("100 iteration"); imshow(ps[:, :, 100])
	subplot(2, 2, 4); title("1000 iteration"); imshow(ps[:, :, 1000])
end

# ╔═╡ 9ffe4f6a-d051-49fc-81ae-50351b492a37
let
	md"""
	Jacobi iteration:
	$(@bind it NumberField(1:1000))
	"""
end

# ╔═╡ d1f7b3c1-bb42-4486-b4a4-fe3e2e4a4765
begin
	pexact(x, y) = sin(2π*x)*sin(2π*y)
	
	figure(size=(700, 300))
	subplot(1,2,1); heatmap(ps[:, :, it])
	subplot(1,2,2); heatmap(pexact.(x, y))
end

# ╔═╡ e4075b90-812a-4478-9c39-5b28ca5dce79
let
	import LinearAlgebra: norm
	err = Float64[]
	for i=1:1500
		push!(err, norm(ps[:, :, i] .- pexact.(x, y)))
	end
	figure(size=(500, 200))
	xlabel("iteration")
	ylabel("error")
	loglog(1:1500, err, [it;], [err[it];], "s")
end

# ╔═╡ d99f1178-ecb0-40e7-baf1-028a4c547354
md"""
Jakość rozwiązania może rozczarowywać, jednak jest to wartość *błędu* a nie *wektora residualnego*.

Podobne wyniki osiągają w [NeuralPDE.jl](https://neuralpde.sciml.ai/dev/pinn/poisson/).
"""

# ╔═╡ f6f5d204-d9ed-4c0c-9cf2-a753d27a38f6
function mulconv(img :: Array{Float64, 2}, kernel :: Array{Float64, 2})
	n, m = size(img)
	I = reshape(img, :, 1)
	K = spdiagm(0 => kernel[2,2]*ones(n*m),
			-1   => kernel[2,1]*ones(n*m-1),
			+1   => kernel[2,3]*ones(n*m-1),
			-m   => kernel[1,2]*ones(n*m-m),
			+m   => kernel[3,2]*ones(n*m-m),
			-m-1 => kernel[1,1]*ones(n*m-m-1),
			-m+1 => kernel[1,3]*ones(n*m-m-1),
			+m-1 => kernel[3,1]*ones(n*m-m-1),
			+m+1 => kernel[3,3]*ones(n*m-m-1))
	reshape(K * I, n, m)
end

# ╔═╡ f2eda34f-8e9e-4683-b067-b6a7fcf1b6d2
@inline function im2col(A, n, m)
	M,N = size(A)
	B = Array{eltype(A)}(undef, m*n, (M-m+1)*(N-n+1))
	indx = reshape(1:M*N, M,N)[1:M-m+1,1:N-n+1]
	for (i,value) in enumerate(indx)
		for j = 0:n-1
		@views B[(i-1)*m*n+j*m+1:(i-1)m*n+(j+1)m] = A[value+j*M:value+m-1+j*M]
		end
	end
	return B'
end

# ╔═╡ fb0a3fea-8ca6-47ba-90d6-f51e9b8b51f7
@inline function _im2col(A, n, m)
	B = Array{eltype(A)}(undef, 3*3, n*m)
	for i=1:m, j=1:n
		b = view(A, i:i+2, j:j+2)
		B[:, i*n-m+j] = reshape(b, :)
	end
	return B
end
	

# ╔═╡ 27ee4149-cdd7-4199-bd62-7ba8d1dd124f
function matconv(img :: Array{Float64, 2}, kernel :: Array{Float64, 2})
	n, m = size(img) .- 2
	I = im2col(img, n, m)
	K = reshape(kernel, 1, :)
	reshape(K * I, n, m)
end

# ╔═╡ 3eeac20a-db8c-4465-b3c9-935f94827b6a
function fftconv(img :: Array{Float64, 2}, kernel :: Array{Float64, 2})
	ker = zero(img); ker[1:3, 1:3] .= kernel
	I = fft(img)
	K = fft(ker)
	J = ifft(I .* K)
	return abs.(J)
end

# ╔═╡ e926ad2a-290a-407b-9077-671f0d311d77
let
	img = cameraman[1:512, 1:512]
	kernel = [1/9 1/9 1/9; 1/9 1/9 1/9; 1/9 1/9 1/9]
	
	tmat = @elapsed res = matconv(img, kernel)
	tmul = @elapsed ult = mulconv(img, kernel)
	tfft = @elapsed ima = fftconv(img, kernel)
	tloop= @elapsed ges =    conv(img, kernel)
	
	tmat = round(Int64, 1e3tmat)
	tmul = round(Int64, 1e3tmul)
	tfft = round(Int64, 1e3tfft)
	tloop= round(Int64, 1e3tloop)
	
	figure(size=(680, 200))
	subplot(1, 4, 1); title("mat: $(tmat)ms");   imshow(res[100:200, 200:300])
	subplot(1, 4, 2); title("mul: $(tmul)ms");   imshow(ult[100:200, 200:300])
	subplot(1, 4, 3); title("FFT: $(tfft)ms");   imshow(ima[100:200, 200:300])
	subplot(1, 4, 4); title("conv: $(tloop)ms"); imshow(ges[100:200, 200:300])
end

# ╔═╡ 216a3337-6f0c-437a-a20f-87c9f372757a
let
	kernel = [1/9 1/9 1/9; 1/9 1/9 1/9; 1/9 1/9 1/9]
	
	t = zeros(10, 4)
	s = zeros(10)
	for i=1:10
		img = zeros(50i, 50i)
		t[i, 1] = @elapsed matconv(img, kernel)
		t[i, 2] = @elapsed mulconv(img, kernel)
		t[i, 3] = @elapsed fftconv(img, kernel)
		t[i, 4] = @elapsed    conv(img, kernel)
		s[i]    = 50i
	end
	
	figure(size=(800, 300))
	xlabel("image size [-]"); ylabel("time [s]")
	semilogy(s, t[:, 1], "s-", s, t[:, 2], "s-", s, t[:, 3], "s-", s, t[:, 4], "s-",
			 labels=["im2col", "mul", "FFT", "loops"], location=11)
end

# ╔═╡ 278c2821-4b89-471d-8c7a-c9776f01012f


# ╔═╡ Cell order:
# ╟─33289b77-d488-4e70-9c55-29a5344ed08a
# ╠═9ae9879e-a377-11eb-084e-09139cd358a6
# ╠═ec947bb1-249f-48a9-88d2-9ecb5370888e
# ╠═7474bb15-a601-47c4-b161-9b86248bdce3
# ╟─646ffce2-8c16-4e5f-88e2-c6ef585cac1e
# ╠═4449113d-1849-4b43-a159-823c41030f88
# ╠═475f9ac7-e2ea-4765-a45e-56b6d949fb62
# ╠═81f30430-eef5-427c-8e32-c291a7699967
# ╟─3175d110-fc9b-44cf-9c79-1210be843eb6
# ╠═30f2260c-a6c2-412c-a855-0d756df4561a
# ╟─400061db-f956-4408-97d5-87218a08625f
# ╠═101baac9-8bab-47f8-9f30-a987b5eda720
# ╠═8d186053-3329-42ee-8b3b-df3d4a9cbdbf
# ╟─e86d9240-ac1f-422c-a5ee-e5132dd651d7
# ╟─f6d2e60f-88e0-47c9-9126-fd2c4ffd6186
# ╠═94b67c4a-5f79-4aca-8fbc-98ad3df573a1
# ╠═b9388c50-7ea3-4b45-9206-855750840b1b
# ╟─10b25830-6bc8-4f25-937a-385330ed95d6
# ╠═ee55200c-1ee1-49ed-83f2-579dfbe97063
# ╟─94dad3ac-62fb-4eb2-9859-c5fc8e44e520
# ╠═136147d8-aab2-446d-ba76-72662f992144
# ╟─5073dc2c-7d4f-42bb-9ff8-d83d3e708185
# ╠═6d623f99-9f2e-4568-9348-0f4e1101fcb7
# ╟─919fe04a-e442-4b05-8641-74e2015ea58e
# ╠═fe58460c-4e71-486d-ab4d-b7b3d9e683d6
# ╠═8a0dc096-0c8c-48e8-99e0-b978d359f6a4
# ╠═68d9573f-8cf0-4b2d-a237-0de3cf7e84c0
# ╟─c1d47a9e-ba03-42e2-9fc9-8c6c8e5bfefd
# ╠═4743da0e-b2a1-4b5f-b2ae-bcd5cde62b80
# ╠═ee0a8cc5-5c55-4925-a48b-f89aa1aeb69f
# ╟─9ffe4f6a-d051-49fc-81ae-50351b492a37
# ╠═d1f7b3c1-bb42-4486-b4a4-fe3e2e4a4765
# ╠═e4075b90-812a-4478-9c39-5b28ca5dce79
# ╟─d99f1178-ecb0-40e7-baf1-028a4c547354
# ╠═b3e17c32-8850-43ac-ae58-565bb553f4b8
# ╠═f6f5d204-d9ed-4c0c-9cf2-a753d27a38f6
# ╠═f2eda34f-8e9e-4683-b067-b6a7fcf1b6d2
# ╠═fb0a3fea-8ca6-47ba-90d6-f51e9b8b51f7
# ╠═27ee4149-cdd7-4199-bd62-7ba8d1dd124f
# ╠═3eeac20a-db8c-4465-b3c9-935f94827b6a
# ╠═e926ad2a-290a-407b-9077-671f0d311d77
# ╠═216a3337-6f0c-437a-a20f-87c9f372757a
# ╠═278c2821-4b89-471d-8c7a-c9776f01012f
