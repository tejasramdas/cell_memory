begin
	using Catalyst, OrdinaryDiffEq, Debugger
	using JumpProcesses
	using StochasticDiffEq, DifferentialEquations
	using WGLMakie
	using DifferentialEquations.EnsembleAnalysis
	using Bonito
	using StatsBase
	using Revise
	using Dates
	using Latexify
end

md"""
# Cellular memory
#### $(Dates.today())
"""

begin
	WGLMakie.activate!()
	Page()
end

includet("functions.jl")
includet("models.jl")

md"""
## Simple switch
"""


begin
    sim_t=2000.0
    t_on=500.0
    t_off=600.0
    l_on = 0.1
    stim=[:t_on => t_on, :t_off => t_off, :l => 0, :l_on => l_on]
end


md"""
### Single switch
"""
simple_switch=Dict()
simple_switch["crn"]=simple_switch_crn
begin
	simple_switch["ts"]=(0.,sim_t)
	simple_switch["u0"]=[:M0 => 1, :M1 => 0]
	simple_switch["ps"]=vcat([:k_f1 => 0.0005,:k_b1 => 0.01],stim)
  simple_switch["models"]=make_models(simple_switch)
end

# simple_switch["sols"]=solve_all(simple_switch;trajectories=100);
# simple_switch["plots"]=Dict()
# simple_switch["plots"]["single"]=make_single_plot(simple_switch)
# simple_switch["plots"]["sde_ens"]=make_ensemble_plot(simple_switch,"sde_ens";bin_width=0.1,step=0.1)
# simple_switch["plots"]["jump_ens"]=make_ensemble_plot(simple_switch,"jump_ens";step=0.1)
# simple_switch["kld"]=Dict()
# simple_switch["kld"]["sde_ens"]=calculate_kld(simple_switch,"sde_ens";rt=t_on-10,step=0.1,bin_width=0.1,xlim="auto")
# simple_switch["kld"]["jump_ens"]=calculate_kld(simple_switch,"jump_ens";rt=t_on-10,step=0.1,xlim="auto")
# simple_switch["plots"]["sde_kld"]=plot_kld(simple_switch,"sde_ens";bin_width=0.1,step=0.1)
# simple_switch["plots"]["jump_kld"]=plot_kld(simple_switch,"jump_ens";bin_width=1,step=0.1)

compute_all!(simple_switch;trajectories=100)


multi_switch=Dict()

for i in 0:5
  multi_switch[4^i]=remake_models(simple_switch,u0=[:M0 => 4^i, :M1 => 0])
  compute_all!(multi_switch[4^i];trajectories=100)
end

md"""
### Multiple switches
"""


# ╔═╡ d3184722-630a-4d81-a186-7b747ffc9914
md"""
## Crick switches
"""


md"""
### Single switch
"""
crick_switch=Dict()
crick_switch["crn"]=crick_crn
begin
	crick_switch["ts"]=(0.,sim_t)
	crick_switch["u0"]=[:M0 => 1, :M1 => 0, :M2=>0]
	crick_switch["ps"]=vcat([:k_f1 => 0.0005,:k_b1 => 0.01,:k_f2 => 0.05, :k_b2 =>0.01],stim)
  crick_switch["models"]=make_models(crick_switch)
end

compute_all!(crick_switch;trajectories=100)


multi_crick=Dict()

for i in 0:5
    @info "N=$(4^i)"
    multi_crick[4^i]=remake_models(crick_switch,u0=[:M0 => 4^i, :M1 => 0, :M2 => 0]);
    compute_all!(multi_crick[4^i];trajectories=100,kld=false)
end

md"""
## Multi-timescale switch
"""

# ╔═╡ c881aade-168f-43e6-840b-6f4ffd23b2d4
md"""
Here we have a four-stage switch where each species triggers the production of the following species rather than being converted to the next species. Every species should get direct input from the stimulus as a control. From a mass-action point of view, the main difference between one species governing the rate of production of the next species (this model) vs. getting converted to the next species (Crick switch) is that the quantity of the first species does not decrease as the subsequent species gets produced, and therefore the timescale of each level is longer.
"""

# ╔═╡ 988edb53-2f7a-4aad-9720-8d4a45f32adc

mts=Dict()
mts["crn"]=mts_crn

stim[4]=(:l_on => 0)

coef=0.05
begin
	mts["ts"]=(0.,1000.)
	mts["u0"]=[:M1 => 0,:M2 => 0, :M3 =>0]
  mts["ps"]=vcat([[[Symbol("k_f$i") => 0.1(coef^i), Symbol("k_b$(i)") => (coef^i)] for i in 1:3]..., stim, [:k_f12 => 0.01,:k_f13 => 0.0, :k_f23 => 0.0]]...)
  mts["models"] = make_models(mts)
end

compute_all!(mts;trajectories=100,kld=false,ensemble=false)

mts["plots"]["single"]


md"""
## CREB
"""

# ╔═╡ c2a5f9f9-fbbb-4f64-807b-611a6c67e3cb

creb=Dict()
creb["crn"]=creb_crn

stim[4]=(:l_on => 10)

begin
    creb["ts"]=(0.,1000.)
    creb["u0"] = [:C1 => 4, :C2 => 0]
    creb["ps"] = [:V_x => 0.1, :V_y => 0.01, :K_x => 5, :K_y => 10, :k_dx => 0.04, :k_dy => 0.01, :r_bas_x => 0.003, :r_bas_y => 0.002, :Ω => 10, :t_on => 10, :t_off => 10, :U_on => 0, :U => 0]
    creb["models"] = make_models(creb)
end

compute_all!(creb;trajectories=100,kld=false,ensemble=true)

creb["plots"]["single"]

creb["plots"]["jump_ens"]

# ╔═╡ d429614c-678d-4545-9364-65a1f8373447
begin
	ts_creb=(0.,1000.)
	u0_creb = [:C1 => 0, :C2 => 0]
	ps_creb = [:V_x => 0.1, :V_y => 0.01, :K_x => 5, :K_y => 10, :k_dx => 0.04, :k_dy => 0.01, :r_bas_x => 0.003, :r_bas_y => 0.002, :Ω => 10]
	creb_models=make_models(creb, u0_creb, ts_creb, ps_creb)
end;

# ╔═╡ 8a8ddc96-4105-4169-8a26-ef13e3abfd3b
sol_creb=solve_all(creb_models;t_on=120,t_off=170);

# ╔═╡ a325f90e-572e-403d-a0f2-00cf7d757de5
make_single_plot(sol_creb,ts_creb,u0_creb;ylim="auto")

# ╔═╡ 50bc615e-fa0d-438f-acef-7bc31d666fcb
make_ensemble_plot(sol_creb["sde_ens"],ts_creb,u0_creb,step=1,ylim="auto")

# ╔═╡ c4e111e0-0917-4a63-80eb-ba8c9bf8c283
make_ensemble_plot(sol_creb["jump_ens"],ts_creb,u0_creb,step=1,ylim="auto")

# ╔═╡ 6af5ce8b-32f1-4e86-9ba7-ca0a4da7ade5
md"""
## Erk
"""

# ╔═╡ 732a8e02-0c7b-4dc6-931a-5adb13b9e60e
erk["crn"]

# ╔═╡ bf582d62-f70c-4fed-bfe8-89a7cf221e55
begin
	U_on = 10.0
	stim_erk=[:t_on => t_on, :t_off => t_off, :U_on => U_on]
end

# ╔═╡ 7327fe06-23f2-4d02-a11a-dec1ba0fa429
begin
	erk["ts"]=(0.,300.)
	erk["u0"] = [:E => 0, :M => 0, :P=>0, :X=>0]
	erk["ps"] = vcat([:K => 0.05, :δ=>50, :γ=>1, :U=>0.0],stim_erk)
	merge!(erk,make_models(erk))
end

# ╔═╡ d0bd6975-d1b8-46d5-9546-7be381e3d2d1
erk["sols"]=solve_all(erk;t_on=120,t_off=170,ensemble=false);

# ╔═╡ 1bdb5d04-c789-42a9-be1b-3559e64fcfbf
make_single_plot(sol_erk,ts_erk,u0_erk;ylim="auto")

# ╔═╡ ce22200e-bf35-4db9-8dc9-1148f6756ae3
md"""
## PKM
"""

# ╔═╡ 78ed549a-7049-491b-811c-84af638dd993
pkm["crn"]

# ╔═╡ 9515ec4a-3117-4c01-a3b9-577674dbeca9
begin
	ts_pkm=(0.,1000.)
	u0_pkm = [:PKM => 0.997, :PKMa => 0.003, :FActin=>1, :FActina=>0, :RNA=>1, :RNAa=>0]
	u0_int_pkm = [:PKM => 997, :PKMa => 3, :FActin=>1000, :FActina=>0, :RNA=>1000, :RNAa=>0]
	ps_pkm=[:j_1 => 10, :j_2 => 0.05, :j_3 => 0.5, :j_4 => 0.16,:τ_1=>1500,:τ_2 =>0.5,:τ_3 => 50,:U=>0]
	pkm_models=make_models(pkm, u0_pkm, ts_pkm, ps_pkm; u0_int=u0_int_pkm)
end;

# ╔═╡ 7bcc9337-da19-4304-a3b7-5ce92545fc82
sol_pkm=solve_all(pkm_models;t_on=120,t_off=170,ensemble=false);

# ╔═╡ bc58d9f0-0106-4d08-8440-15ea3e8ef12e
make_single_plot(sol_pkm,ts_pkm,u0_pkm;ylim="auto",idxs=[2,4,6])

# ╔═╡ 56bdd0db-a1b6-48d5-a668-8b0d10aa96c2
md"""
## Epigenetic
"""

# ╔═╡ 2ebe8946-704b-48bb-9959-7bfb5d4f6d69
epi_simple

# ╔═╡ 02c4dc51-35a3-4e7b-9221-562578e510e7
md"""
## End
"""

# ╔═╡ 4175fde0-0549-4545-990d-3d79f594f9c9
#=╠═╡
begin
	f=Figure(size=(1500,500))
	ax=[Axis(f[1,i]) for i in 1:3]
	f
end
  ╠═╡ =#

# ╔═╡ 7f12d6e0-0e8e-49a9-82e8-4ddc2f9f8bfd
# ╠═╡ disabled = true
#=╠═╡
begin
	f=Figure(size=(1000,400))
	ax1=Axis(f[1:5,1:5])
	ax2=Axis(f[1:5,6:10])
	slid=Makie.Slider(f[6,:],range=1:10:1000)
	lines!(ax1,0:0.1:300,@lift(x[$(slid.value)].(0:0.1:300;idxs=2)))
	lines!(ax2,0:0.1:300,@lift(y[$(slid.value)].(0:0.1:300;idxs=2)))
	ylims!(ax1,0,100)
	ylims!(ax2,0,100)
	f
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─6518c587-8638-4909-a995-b331634c8079
# ╠═9cbb14a7-884b-4783-8fb4-650260d940d4
# ╟─524af17b-ef72-4a03-a471-585db57f3ec1
# ╠═80447764-5654-4235-87b5-f447e68a8438
# ╠═ba10e90c-4ba2-4414-85d5-7174c7ad17df
# ╟─daddeeae-c4c1-4028-80c4-9685023c2aad
# ╠═c8642392-804b-40f5-be7f-9474226fc9e7
# ╟─c424dc2f-3a01-4d7e-89c7-663a6f2deed5
# ╠═53b51591-cd8e-47bf-b6f0-0a943da78add
# ╠═7100aae4-aabd-46f6-b0bf-f2135c6aa750
# ╠═c21493cc-12fd-4e67-a914-9567bb81d033
# ╠═c419430c-decd-4cb9-b73d-8a067581e77b
# ╠═53297ff5-600e-40f9-8c66-1a606655f366
# ╠═fae1223c-3eec-42ee-94d7-2851e3e9781c
# ╠═b5093fa8-3d6d-4b87-ae45-25ffae5bfbd7
# ╠═4e82bb1c-cda3-4a98-adfd-e7a2e656b798
# ╠═cffc6d07-4eec-4a75-9a95-551eeb75db81
# ╠═6196f720-6d00-45a1-a89a-d864eee9a82a
# ╠═05636dd8-517a-4b50-92e5-22fbd0e79241
# ╠═b52e17c7-f6dc-4711-9144-cc29355db361
# ╠═a7f452c5-067c-4c42-b6ec-a9439ff94d06
# ╠═b8103024-6fc2-49b1-b3c1-c1201ebcdc65
# ╠═9870b176-cca9-4de7-9931-2b3660133c2e
# ╠═09d05769-285a-4903-abe1-c8534c061ca0
# ╠═1c34bd46-c116-4ec4-9fbb-9a09681d7b7a
# ╟─ca515e68-f9fa-4ccc-adf9-e8de8f4905d3
# ╟─f7232788-9741-4f5d-8790-0c79671c876a
# ╠═0af77422-a512-4e60-8620-eea173bdb08f
# ╠═662fbc83-03d9-4c91-a359-1c6c36365988
# ╠═060c4219-33c9-4fe9-ba65-e412668eb31d
# ╠═5b781245-497e-423e-b4a8-e02d3ea90c73
# ╠═69c0612e-30f4-445f-8c9f-8589d5313502
# ╠═ce3407d1-c524-43a6-830d-fbba078bbc75
# ╠═6f7c6b6f-195e-4283-905f-b956bed191ab
# ╠═a382e5b6-abd4-48b1-bfed-be2793e20bb0
# ╠═75527d17-bb0d-46dd-a462-6ab166b5bd82
# ╠═544cdc7c-19b7-4cf5-b1ad-242aa893fb23
# ╠═cd1175f0-136b-46de-92d7-cfbd2ecc4987
# ╠═ac305374-c773-4359-b2b5-459112c65460
# ╠═7f8a93fc-07f9-4ec5-918d-ff4dc7f40f27
# ╠═a02e7f1b-9877-4f6c-8e9b-1bd9eed163fb
# ╠═c5e6a9ce-e9dc-4946-9fad-89e120763fc4
# ╠═b4324056-3d0e-4ffd-ac21-923c8444a5a5
# ╠═4175fde0-0549-4545-990d-3d79f594f9c9
# ╠═a235e015-8518-493f-8a8c-d651bd7e59b9
# ╠═bb802d84-2b08-4a7a-8378-21b3fc024c71
# ╟─f5cc3e6d-73cf-4c23-a075-31008153db3b
# ╠═5d15ba23-3744-40dd-b350-7a2eddfac4ad
# ╠═0b3c11d0-6f48-499c-9b57-72c0ea2edf40
# ╠═9f9d5a5e-7b44-470a-8d9f-7e3182afed9d
# ╠═b75b0a55-a1f9-4c3c-8cf2-45159c67b669
# ╠═7f12d6e0-0e8e-49a9-82e8-4ddc2f9f8bfd
# ╠═770360a9-bd9e-405d-af35-9f07f871c599
# ╠═dcb5cb12-8e79-48a6-9320-1348fb92a330
# ╠═8f23cd47-8cc4-4c5b-ba64-afc45ff274d8
# ╠═755ef828-e5dd-4b83-9272-bd396c898d4d
# ╠═a03d42f1-fe47-4366-ad74-c0dcaed958a5
# ╠═e48fa896-6e40-4bcf-9d80-b97dca7b1951
# ╠═d198d116-e803-4f3f-9814-4fe5bc4b5bd8
# ╠═6087db7a-9190-4133-9bd9-706ed44ff00a
# ╠═d3184722-630a-4d81-a186-7b747ffc9914
# ╠═3ed7b1d6-d4be-4500-8a51-af57bc2641a9
# ╠═2364af6c-d6d6-4045-9441-c494ccb66981
# ╠═2f381ec5-2fd3-4a87-8dfd-5239bf9ed8b5
# ╠═edf776ba-23db-4240-9a69-4c2d738f7e07
# ╠═0e6119f4-0f1f-49dd-8e87-ca26b86d3f68
# ╠═ee4b9acd-305d-41c4-928b-316e149942b8
# ╠═89814c74-6a4d-4f57-a384-dd07697a39fc
# ╠═24ef355a-0e28-4512-b7df-b7214c38070f
# ╠═6fbf7de8-bd5f-4c43-9580-a7d47c613003
# ╠═f53958f8-dd41-41f1-bc4c-134339998a24
# ╟─262b2285-5013-4102-a4d9-4ae262e0ebe6
# ╟─7119430a-6938-4e41-9ece-463c995e04e3
# ╠═c881aade-168f-43e6-840b-6f4ffd23b2d4
# ╠═988edb53-2f7a-4aad-9720-8d4a45f32adc
# ╠═17aacbe0-0f5e-4115-8407-e0547832c18f
# ╠═95d4e4c8-ac84-4caa-b961-f41b6fc52d8f
# ╠═b9dd6e51-9e95-4aae-93d4-b23b2982a763
# ╠═817e4d6f-561a-44e8-bf56-eeb8f40ddf2e
# ╠═5b8c14ad-2aeb-405b-9f55-bfa0783fa8ec
# ╠═8217284d-0f43-40e6-a434-aa3d0ad45b30
# ╟─46c2e1c8-565b-4f2a-8b42-ff34b0b83c90
# ╠═c2a5f9f9-fbbb-4f64-807b-611a6c67e3cb
# ╠═d429614c-678d-4545-9364-65a1f8373447
# ╠═8a8ddc96-4105-4169-8a26-ef13e3abfd3b
# ╠═a325f90e-572e-403d-a0f2-00cf7d757de5
# ╠═50bc615e-fa0d-438f-acef-7bc31d666fcb
# ╠═c4e111e0-0917-4a63-80eb-ba8c9bf8c283
# ╠═6af5ce8b-32f1-4e86-9ba7-ca0a4da7ade5
# ╠═732a8e02-0c7b-4dc6-931a-5adb13b9e60e
# ╠═bf582d62-f70c-4fed-bfe8-89a7cf221e55
# ╠═7327fe06-23f2-4d02-a11a-dec1ba0fa429
# ╠═d0bd6975-d1b8-46d5-9546-7be381e3d2d1
# ╠═1bdb5d04-c789-42a9-be1b-3559e64fcfbf
# ╠═ce22200e-bf35-4db9-8dc9-1148f6756ae3
# ╠═78ed549a-7049-491b-811c-84af638dd993
# ╠═9515ec4a-3117-4c01-a3b9-577674dbeca9
# ╠═7bcc9337-da19-4304-a3b7-5ce92545fc82
# ╠═bc58d9f0-0106-4d08-8440-15ea3e8ef12e
# ╠═56bdd0db-a1b6-48d5-a668-8b0d10aa96c2
# ╠═2ebe8946-704b-48bb-9959-7bfb5d4f6d69
# ╠═02c4dc51-35a3-4e7b-9221-562578e510e7
