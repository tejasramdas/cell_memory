### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 80447764-5654-4235-87b5-f447e68a8438
begin
	using Pkg
	Pkg.activate(".")
	using Catalyst, OrdinaryDiffEq, Debugger
	using JumpProcesses
	using StochasticDiffEq, DifferentialEquations
	using WGLMakie
	using DifferentialEquations.EnsembleAnalysis
	using Bonito
	using InformationMeasures, StatsBase
	using Revise
end

# ╔═╡ d128e5d3-969e-497c-95c7-6513717760fb
include("helper.jl")

# ╔═╡ 6518c587-8638-4909-a995-b331634c8079
md"""
# Cellular memory
"""

# ╔═╡ daddeeae-c4c1-4028-80c4-9685023c2aad
md"""
## Load packages
"""

# ╔═╡ 9cbb14a7-884b-4783-8fb4-650260d940d4
html"""
<style>main {max-width: 1000px;}</style>
<style>pluto-output.scroll_y {max-height: 2000px;}</style>
"""

# ╔═╡ c8642392-804b-40f5-be7f-9474226fc9e7
begin
	WGLMakie.activate!()
	Page()
end

# ╔═╡ ca515e68-f9fa-4ccc-adf9-e8de8f4905d3
md"""
## All models
"""

# ╔═╡ f7232788-9741-4f5d-8790-0c79671c876a


# ╔═╡ e114fdb6-b0ef-4855-9b96-2b121fbdffd3
simple_switch = @reaction_network begin
	@parameters t_on t_off l_on
	@default_noise_scaling 0.2
	@discrete_events begin
		((t == t_on)) => [l ~ l_on]
		((t == t_off)) => [l ~ 0.0]
	end
	(k_f1*(10*l+1),k_b1), M0 <--> M1
end

# ╔═╡ 036d180c-14e8-41e4-ba40-a84477e82f15
begin
simple_binary = @reaction_network begin
	@default_noise_scaling 0.2
	@discrete_events begin
		((t == 20)) => [l ~ 1.0]
		((t == 70)) => [l ~ 0.0]
	end
	(k_f*(10*l+1), k_b), M0 <--> M1
end
end

# ╔═╡ 0af77422-a512-4e60-8620-eea173bdb08f
crick = @reaction_network begin
	@parameters t_on t_off l_on
	@default_noise_scaling 0.2
	@discrete_events begin
		((t == t_on)) => [l ~ l_on]
		((t == t_off)) => [l ~ 0.0]
	end
	(k_f1*(10*l+1),k_b1), M0 <--> M1
	(k_f2,k_b2), M1 <--> M2
end

# ╔═╡ 662fbc83-03d9-4c91-a359-1c6c36365988
mts = @reaction_network begin
	@parameters t_on t_off l_on
	@default_noise_scaling 0.2
	@discrete_events begin
		((t == t_on)) => [l ~ l_on]
		((t == t_off)) => [l ~ 0.0]
	end
	(k_f1*(100*l+1),k_b1), M1 <--> M1a
	(k_f2*(0*l+10*M1a+1),k_b2), M2 <--> M2a
	(k_f3*(0*l+10*M2a+1),k_b3), M3 <--> M3a
	(k_f4*(0*l+10*M3a+1),k_b4), M4 <--> M4a
end

# ╔═╡ 060c4219-33c9-4fe9-ba65-e412668eb31d
begin
	hillcreb(C1,C2,K_x,K_y,V,Ω) = hillar(C1/(√K_x *Ω),C2/(√K_y *Ω), V*Ω, 1, 2)
	creb = @reaction_network begin
		@default_noise_scaling 0.2
	    (Ω*r_bas_x, Ω*r_bas_y), 0 --> (C1, C2)
	    (k_dx, k_dy), (C1, C2) --> 0
	    (hillcreb(C1,C2,K_x,K_y,V_x,Ω), hillcreb(C1,C2,K_x,K_y,V_y,Ω)), 0 --> (C1,C2)
	end
end

# ╔═╡ 5b781245-497e-423e-b4a8-e02d3ea90c73
erk = @reaction_network begin
	@parameters t_on t_off U_on
    @default_noise_scaling 0.2
	@discrete_events begin
        ((t == t_on)) => [U ~ U_on]
        ((t == t_off)) => [U ~ 0.0]
    end
    (U,hillar(E,0,1,K,2),M,hillar(E,0,1,K,2)), 0 --> (E,M,P,X)
    (γ+δ*P,γ,γ,γ), (E,M,P,X) --> 0
end

# ╔═╡ 69c0612e-30f4-445f-8c9f-8589d5313502
pkm = @reaction_network begin
	@default_noise_scaling 0.2
    @discrete_events begin
        ((t == 20)) => [U ~ 10.0]
        ((t == 70)) => [U ~ 0.0]
    end
    (j_1*RNA/τ_1,1/τ_1), PKM <--> PKMa
    (j_2+j_3*PKMa/τ_2,1/τ_2), FActin <--> FActina
    (j_4*FActina*(PKMa+U)/τ_3,1/τ_3) ,RNA <--> RNAa
end

# ╔═╡ 75527d17-bb0d-46dd-a462-6ab166b5bd82
md"""
## Simple switch
"""

# ╔═╡ cd1175f0-136b-46de-92d7-cfbd2ecc4987
simple_switch

# ╔═╡ 7f8a93fc-07f9-4ec5-918d-ff4dc7f40f27
begin
	t_on=120.0
	t_off=170.0
	l_on = 1
	stim=[:t_on => t_on, :t_off => t_off, :l => 0, :l_on => l_on]
end

# ╔═╡ f5cc3e6d-73cf-4c23-a075-31008153db3b
md"""
### Single switch
"""

# ╔═╡ bc97d259-e6a7-42d8-80f7-f6e29e3cdfa0
begin
	ts_simple=(0.,300.)
	u0_simple=[:M0 => 1, :M1 => 0]
	ps_simple=vcat([:k_f1 => 0.005,:k_b1 => 0.1],stim)
	simple_models=make_models(simple_switch, u0_simple, ts_simple, ps_simple)
end

# ╔═╡ eaa47f4a-9d83-4f73-9076-363ea84f3f45
sol_simple=solve_all(simple_models;t_on=120,t_off=170);

# ╔═╡ 5757f358-12bf-4040-8437-fa0171b5f032
make_single_plot(sol_simple,ts_simple,u0_simple)

# ╔═╡ fae1bed4-d76b-4c4b-8456-499feac5d225
make_ensemble_plot(sol_simple["sde_ens"],ts_simple,u0_simple)

# ╔═╡ e1d6348c-8c0b-421b-987d-7b5f1afd91fd
make_ensemble_plot(sol_simple["jump_ens"],ts_simple,u0_simple)

# ╔═╡ 71c67272-eab2-43e8-a2eb-5c3cb484057b
kld_simple=calculate_kld(sol_simple["jump_ens"],get_ts(ts_simple),u0_simple,rt=110)

# ╔═╡ 1af868b4-51ba-4dd5-bedf-4a970a1bcbc7
plot_kld(kld_simple,get_ts(ts_simple))

# ╔═╡ dcb5cb12-8e79-48a6-9320-1348fb92a330
md"""
### Multiple switches
"""

# ╔═╡ 9f9d5a5e-7b44-470a-8d9f-7e3182afed9d
# begin
# 	ts_simple=(0.,300.)
# 	u0_simple=[:M0 => 1, :M1 => 0]
# 	ps_simple=vcat([:k_f1 => 0.005,:k_b1 => 0.1],stim)
# 	simple_models=make_models(simple_switch, u0_simple, ts_simple, ps_simple)
# end

# ╔═╡ 8f23cd47-8cc4-4c5b-ba64-afc45ff274d8
# sol_simple=solve_all(simple_models;t_on=120,t_off=170);

# ╔═╡ 755ef828-e5dd-4b83-9272-bd396c898d4d
# make_single_plot(sol_simple,ts_simple,u0_simple)

# ╔═╡ a03d42f1-fe47-4366-ad74-c0dcaed958a5
# make_ensemble_plot(sol_simple["sde_ens"],ts_simple,u0_simple)

# ╔═╡ e48fa896-6e40-4bcf-9d80-b97dca7b1951
# make_ensemble_plot(sol_simple["jump_ens"],ts_simple,u0_simple)

# ╔═╡ d3184722-630a-4d81-a186-7b747ffc9914
md"""
## Crick switches
"""

# ╔═╡ 3ed7b1d6-d4be-4500-8a51-af57bc2641a9
crick

# ╔═╡ 89fc758b-f492-4686-a962-162bd289aa76
ts_crick=(0.,300.)

# ╔═╡ 4c5b7ba7-d10b-4949-ac65-c95857cb0889
begin
	u0_crick=[:M0 => 10, :M1 => 0, :M2 => 0]
	ps_crick=vcat([:k_f1 => 0.005,:k_b1 => 0.1,:k_f2 => 0.05,:k_b2 => 0.1], stim)
	crick_models=make_models(crick, u0_crick, ts_crick, ps_crick)
end

# ╔═╡ 218a3905-83ae-457d-a6d1-d41f93b2a3d4
sol_crick=solve_all(crick_models;t_on=120,t_off=170);

# ╔═╡ 9d702f99-d9c6-4ba9-a419-570e858add75
make_single_plot(sol_crick,ts_crick,u0_crick)

# ╔═╡ b98c1089-8898-4040-a7d8-9d515a4fa984
make_ensemble_plot(sol_crick["sde_ens"],ts_crick,u0_crick)

# ╔═╡ 659484ab-fbde-4728-ba28-40f6fd7054f6
make_ensemble_plot(sol_crick["jump_ens"],ts_crick,u0_crick)

# ╔═╡ eaea440c-e04c-45ed-8253-fa294bc19f53
kld_crick=calculate_kld(sol_crick["jump_ens"],get_ts(ts_crick),u0_crick,rt=110)

# ╔═╡ eb5e987d-419d-4b13-8780-2e6aecc49648
plot_kld(kld_crick,get_ts(ts_crick))

# ╔═╡ 9e902065-8186-431d-85b3-4ca7c6cf4b81
@time make_ensemble_plot(sol_crick["jump_ens"],ts_crick,u0_crick)

# ╔═╡ 262b2285-5013-4102-a4d9-4ae262e0ebe6
md"""
### Crick switch scaling
"""

# ╔═╡ 7119430a-6938-4e41-9ece-463c995e04e3
md"""
## Multi-timescale switch
"""

# ╔═╡ c881aade-168f-43e6-840b-6f4ffd23b2d4
md"""
Here we have a four-stage switch where each species triggers the production of the following species rather than being converted to the next species. Every species should also get direct  From a mass-action point of view, The main difference is what the base rate of each species is. 
"""

# ╔═╡ 988edb53-2f7a-4aad-9720-8d4a45f32adc
mts

# ╔═╡ 17aacbe0-0f5e-4115-8407-e0547832c18f
begin
	ts_mts=(0.,1000.)
	u0_mts=vcat([[Symbol("M$i") => 10, Symbol("M$(i)a") => 0] for i in 1:4]...)
	ps_mts=vcat([[[Symbol("k_f$i") => 0.01*(0.2^i), Symbol("k_b$(i)") => 0.2^i] for i in 1:4]..., stim]...)
	mts_models=make_models(mts, u0_mts, ts_mts, ps_mts)
end

# ╔═╡ 95d4e4c8-ac84-4caa-b961-f41b6fc52d8f
sol_mts=solve_all(mts_models;t_on=120,t_off=170);

# ╔═╡ b9dd6e51-9e95-4aae-93d4-b23b2982a763
make_single_plot(sol_mts,ts_mts,u0_mts;idxs=[2,4,6,8])

# ╔═╡ 817e4d6f-561a-44e8-bf56-eeb8f40ddf2e
make_ensemble_plot(sol_mts["sde_ens"],ts_mts,u0_mts;idxs=[2,4,6,8])

# ╔═╡ 5b8c14ad-2aeb-405b-9f55-bfa0783fa8ec
make_ensemble_plot(sol_mts["jump_ens"],ts_mts,u0_mts;idxs=[2,4,6,8])

# ╔═╡ 37f40d13-e7cd-47cb-83eb-215320435a4f


# ╔═╡ 46c2e1c8-565b-4f2a-8b42-ff34b0b83c90
md"""
## CREB
"""

# ╔═╡ c2a5f9f9-fbbb-4f64-807b-611a6c67e3cb
creb

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
make_ensemble_plot(sol_creb["sde_ens"],ts_creb,u0_creb)

# ╔═╡ c4e111e0-0917-4a63-80eb-ba8c9bf8c283
make_ensemble_plot(sol_creb["jump_ens"],ts_creb,u0_creb)

# ╔═╡ 14cb190a-3cb1-4ba4-b496-b8f86bf9fe0e


# ╔═╡ 6af5ce8b-32f1-4e86-9ba7-ca0a4da7ade5
md"""
## Erk
"""

# ╔═╡ 732a8e02-0c7b-4dc6-931a-5adb13b9e60e
erk

# ╔═╡ bf582d62-f70c-4fed-bfe8-89a7cf221e55
begin
	U_on = 10.0
	stim_erk=[:t_on => t_on, :t_off => t_off, :U_on => U_on]
end

# ╔═╡ 7327fe06-23f2-4d02-a11a-dec1ba0fa429
begin
	ts_erk=(0.,300.)
	u0_erk = [:E => 0, :M => 0, :P=>0, :X=>0]
	ps_erk = vcat([:K => 0.05, :δ=>50, :γ=>1, :U=>0.0],stim_erk)
	erk_models=make_models(erk, u0_erk, ts_erk, ps_erk)
end

# ╔═╡ d0bd6975-d1b8-46d5-9546-7be381e3d2d1
sol_erk=solve_all(erk_models;t_on=120,t_off=170,ensemble=false);

# ╔═╡ 1bdb5d04-c789-42a9-be1b-3559e64fcfbf
make_single_plot(sol_erk,ts_erk,u0_erk;ylim="auto")

# ╔═╡ ce22200e-bf35-4db9-8dc9-1148f6756ae3
md"""
## PKM
"""

# ╔═╡ 78ed549a-7049-491b-811c-84af638dd993
pkm

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

# ╔═╡ 02c4dc51-35a3-4e7b-9221-562578e510e7
md"""
## End
"""

# ╔═╡ Cell order:
# ╟─6518c587-8638-4909-a995-b331634c8079
# ╠═80447764-5654-4235-87b5-f447e68a8438
# ╟─daddeeae-c4c1-4028-80c4-9685023c2aad
# ╠═9cbb14a7-884b-4783-8fb4-650260d940d4
# ╠═d128e5d3-969e-497c-95c7-6513717760fb
# ╠═c8642392-804b-40f5-be7f-9474226fc9e7
# ╟─ca515e68-f9fa-4ccc-adf9-e8de8f4905d3
# ╟─f7232788-9741-4f5d-8790-0c79671c876a
# ╠═e114fdb6-b0ef-4855-9b96-2b121fbdffd3
# ╟─036d180c-14e8-41e4-ba40-a84477e82f15
# ╠═0af77422-a512-4e60-8620-eea173bdb08f
# ╠═662fbc83-03d9-4c91-a359-1c6c36365988
# ╠═060c4219-33c9-4fe9-ba65-e412668eb31d
# ╠═5b781245-497e-423e-b4a8-e02d3ea90c73
# ╠═69c0612e-30f4-445f-8c9f-8589d5313502
# ╠═75527d17-bb0d-46dd-a462-6ab166b5bd82
# ╠═cd1175f0-136b-46de-92d7-cfbd2ecc4987
# ╠═7f8a93fc-07f9-4ec5-918d-ff4dc7f40f27
# ╟─f5cc3e6d-73cf-4c23-a075-31008153db3b
# ╠═bc97d259-e6a7-42d8-80f7-f6e29e3cdfa0
# ╠═eaa47f4a-9d83-4f73-9076-363ea84f3f45
# ╠═5757f358-12bf-4040-8437-fa0171b5f032
# ╠═fae1bed4-d76b-4c4b-8456-499feac5d225
# ╠═e1d6348c-8c0b-421b-987d-7b5f1afd91fd
# ╠═71c67272-eab2-43e8-a2eb-5c3cb484057b
# ╠═1af868b4-51ba-4dd5-bedf-4a970a1bcbc7
# ╠═dcb5cb12-8e79-48a6-9320-1348fb92a330
# ╠═9f9d5a5e-7b44-470a-8d9f-7e3182afed9d
# ╠═8f23cd47-8cc4-4c5b-ba64-afc45ff274d8
# ╠═755ef828-e5dd-4b83-9272-bd396c898d4d
# ╠═a03d42f1-fe47-4366-ad74-c0dcaed958a5
# ╠═e48fa896-6e40-4bcf-9d80-b97dca7b1951
# ╠═d3184722-630a-4d81-a186-7b747ffc9914
# ╠═3ed7b1d6-d4be-4500-8a51-af57bc2641a9
# ╠═89fc758b-f492-4686-a962-162bd289aa76
# ╠═4c5b7ba7-d10b-4949-ac65-c95857cb0889
# ╠═218a3905-83ae-457d-a6d1-d41f93b2a3d4
# ╠═9d702f99-d9c6-4ba9-a419-570e858add75
# ╠═b98c1089-8898-4040-a7d8-9d515a4fa984
# ╠═659484ab-fbde-4728-ba28-40f6fd7054f6
# ╠═eaea440c-e04c-45ed-8253-fa294bc19f53
# ╠═eb5e987d-419d-4b13-8780-2e6aecc49648
# ╠═9e902065-8186-431d-85b3-4ca7c6cf4b81
# ╟─262b2285-5013-4102-a4d9-4ae262e0ebe6
# ╟─7119430a-6938-4e41-9ece-463c995e04e3
# ╠═c881aade-168f-43e6-840b-6f4ffd23b2d4
# ╠═988edb53-2f7a-4aad-9720-8d4a45f32adc
# ╠═17aacbe0-0f5e-4115-8407-e0547832c18f
# ╠═95d4e4c8-ac84-4caa-b961-f41b6fc52d8f
# ╠═b9dd6e51-9e95-4aae-93d4-b23b2982a763
# ╠═817e4d6f-561a-44e8-bf56-eeb8f40ddf2e
# ╠═5b8c14ad-2aeb-405b-9f55-bfa0783fa8ec
# ╠═37f40d13-e7cd-47cb-83eb-215320435a4f
# ╟─46c2e1c8-565b-4f2a-8b42-ff34b0b83c90
# ╠═c2a5f9f9-fbbb-4f64-807b-611a6c67e3cb
# ╠═d429614c-678d-4545-9364-65a1f8373447
# ╠═8a8ddc96-4105-4169-8a26-ef13e3abfd3b
# ╠═a325f90e-572e-403d-a0f2-00cf7d757de5
# ╠═50bc615e-fa0d-438f-acef-7bc31d666fcb
# ╠═c4e111e0-0917-4a63-80eb-ba8c9bf8c283
# ╠═14cb190a-3cb1-4ba4-b496-b8f86bf9fe0e
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
# ╠═02c4dc51-35a3-4e7b-9221-562578e510e7
