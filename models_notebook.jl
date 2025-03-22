### A Pluto.jl notebook ###
# v0.20.5

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
	using Dates
	using Latexify
end

# ╔═╡ 6518c587-8638-4909-a995-b331634c8079
md"""
# Cellular memory
#### $(Dates.today())
"""

# ╔═╡ 9cbb14a7-884b-4783-8fb4-650260d940d4
html"""
<style>main {max-width: 100%; overflow:hidden; padding-right: 5%;}</style>
<style>pluto-editor main {margin-right: 0;}</style>
<style>pluto-output.scroll_y {max-height: 2000px;}</style>
"""

# ╔═╡ 524af17b-ef72-4a03-a471-585db57f3ec1
md"""
## Activate packages
"""

# ╔═╡ daddeeae-c4c1-4028-80c4-9685023c2aad
md"""
## Load packages
"""

# ╔═╡ c8642392-804b-40f5-be7f-9474226fc9e7
begin
	WGLMakie.activate!()
	Page()
end

# ╔═╡ c424dc2f-3a01-4d7e-89c7-663a6f2deed5
md"""
## Helper functions
"""

# ╔═╡ 53b51591-cd8e-47bf-b6f0-0a943da78add
u_int(u) = map(i->Pair(i[1],Int(i[2])),u)

# ╔═╡ 7100aae4-aabd-46f6-b0bf-f2135c6aa750
function get_ts(ts;step=0.1)
	ts[1]:step:ts[2]
end

# ╔═╡ c21493cc-12fd-4e67-a914-9567bb81d033
function transform_sol(sol,t;idxs=nothing)
	hcat(sol.(t,idxs=idxs)...)
end

# ╔═╡ c419430c-decd-4cb9-b73d-8a067581e77b
function transform_ens_sol_qts(sol,t,qts)
	concat_sol=cat([hcat(timeseries_point_quantile(sol,q,t)...) for q in qts]...,dims=3)
end

# ╔═╡ 53297ff5-600e-40f9-8c66-1a606655f366
function transform_ens_sol(sol,t)
	concat_sol=cat([cat(componentwise_vectors_timepoint(sol,i)...,dims=2) for i in t]...,dims=3)
	permutedims(concat_sol,[2,3,1])
end

# ╔═╡ fae1223c-3eec-42ee-94d7-2851e3e9781c
function transform_ens_sol_mean(sol,t)
	concat_sol_qts=hcat(timeseries_point_mean(sol,t)...)
end

# ╔═╡ b5093fa8-3d6d-4b87-ae45-25ffae5bfbd7
get_labels(u) = map(i->i[1],u)

# ╔═╡ 4e82bb1c-cda3-4a98-adfd-e7a2e656b798
function make_models(model;u0_int=nothing,jump_p=true)
    ode=ODEProblem(model["crn"],model["u0"],model["ts"],model["ps"])
    sde=SDEProblem(model["crn"],model["u0"],model["ts"],model["ps"])
    if jump_p
		if isnothing(u0_int)
			u0_int=u_int(model["u0"])
		end
        jinput = JumpInputs(model["crn"], u0_int, model["ts"],model["ps"])
        jprob = JumpProblem(jinput)
        eprob_jump=EnsembleProblem(jprob)
    else
        jprob=nothing
        eprob=nothing
    end
	eprob_sde = EnsembleProblem(sde)
    return Dict(["ode" => ode,"sde"=> sde, "jump" => jprob, "eprob_sde" => eprob_sde, "eprob_jump" => eprob_jump])
end

# ╔═╡ cffc6d07-4eec-4a75-9a95-551eeb75db81
function solve_all(models;t_on=0,t_off=0,trajectories=1000,ensemble=true)
	tstops=[t_on,t_off]
	sols=map(i->(i=>solve(models[i];tstops=tstops)),["ode","sde","jump"])
	if ensemble
		push!(sols,("sde_ens" => solve(models["eprob_sde"],STrapezoid();tstops=tstops,trajectories=trajectories)))
		push!(sols,("jump_ens" => solve(models["eprob_jump"],SSAStepper();tstops=tstops,trajectories=trajectories)))
	end
	Dict(sols)
end

# ╔═╡ 6196f720-6d00-45a1-a89a-d864eee9a82a
function calculate_kld(sol,sol_type; rt=0,idxs=nothing,bin_width=1,xlim=nothing,step=0.1)
	sol,ts,u0=sol["sols"][sol_type], sol["ts"], sol["u0"]
	if isnothing(idxs)
		idxs=1:size(u0,1)
		i_out=idxs[end]
	end
	kl_divs=[]
	if isnothing(xlim)
		xlim=u0[1][2]
	elseif xlim=="auto"
		xlim=maximum(timeseries_point_quantile(sol, 1, t))
	else
		xlim=xlim
	end
	bins=Tuple([-bin_width:bin_width:xlim+bin_width for idx in idxs])
	h_rt=fit(Histogram, Tuple(componentwise_vectors_timepoint(sol,rt)[idxs]),bins)
	for t in get_ts(ts,step=step)
		h=fit(Histogram, Tuple(componentwise_vectors_timepoint(sol,t)[idxs]),bins)
		push!(kl_divs,kldivergence(h_rt.weights,h.weights))
	end
	kl_divs
end

# ╔═╡ 05636dd8-517a-4b50-92e5-22fbd0e79241
function plot_sol(sol,t,u0;f=Figure(),idxs=nothing,leg=nothing,ylim=nothing)
	if isnothing(idxs)
		idxs=1:size(u0,1)
	end
	ax=Axis(f[1,1])
	plot_sol=transform_sol(sol,t,idxs=idxs)
	l=series!(ax,t,plot_sol,labels=["$(get_labels(u0)[i])" for i in idxs])
	if isnothing(ylim)
		ylims!(ax,-0.5,u0[1][2]+0.5)
	elseif ylim=="auto"
		ylims!(ax,-0.5,2*maximum(plot_sol)+0.5)
	else
		ylims!(ax,-0.5,ylim)
	end
	if !isnothing(leg)
		Legend(leg,ax)
	end
	f
end

# ╔═╡ b52e17c7-f6dc-4711-9144-cc29355db361
function plot_ens_hist(sol,t,u0;bin_width=1,f=Figure(),leg=nothing,idxs=nothing,xlim=nothing)
	if isnothing(idxs)
		idxs=1:size(u0,1)
	end
	if isnothing(xlim)
		xlim=u0[1][2]
	elseif xlim=="auto"
		xlim=maximum(timeseries_point_quantile(sol, 1, t))
	else
		xlim=xlim
	end
	ax=Axis(f[1,1])
	s=Makie.Slider(f[2,1],range = t)
	l=[hist!(ax,@lift(componentwise_vectors_timepoint(sol,$(s.value))[idxs[i]]),label="$(get_labels(u0)[idxs[i]])",bins=(-bin_width/2:bin_width:xlim+bin_width/2)) for i in 1:size(idxs,1)]
	if !isnothing(leg)
		Legend(leg,ax)
	end
	xlims!(ax,-0.5,xlim+0.5)
	title_lab = Label(f[0,1],@lift("Timepoint $($(s.value))"),tellwidth=false)
	f
end

# ╔═╡ a7f452c5-067c-4c42-b6ec-a9439ff94d06
function plot_mean(sol,t,u0;qs=0.25:0.25:1,idxs=nothing,f=Figure(),ylim=nothing,leg=nothing)
	if isnothing(idxs)
		idxs=1:size(u0,1)
	end
	ax=Axis(f[1:5,1])
	plot_sol=transform_ens_sol_mean(sol,t)[idxs,:]
	l=series!(ax,t,plot_sol,labels=["$(get_labels(u0)[i])" for i in idxs])
	xlims!(ax,t[1],t[end])
	if isnothing(ylim)
		ylims!(ax,-0.5,u0[1][2]+0.5)
	elseif ylim=="auto"
		ylims!(ax,-0.5,maximum(plot_sol)+0.5)
	else
		ylims!(ax,-0.5,ylim)
	end
	if !isnothing(leg)
		Legend(leg,ax)
	end
	f
end

# ╔═╡ b8103024-6fc2-49b1-b3c1-c1201ebcdc65
function plot_qt(sol,t,u0;qs=0.25:0.25:1,idxs=nothing,f=Figure(),ylim=nothing,leg=nothing)
	if isnothing(idxs)
		idxs=1:size(u0,1)
	end
	axs=[Axis(f[1:5,i]) for i in 1:size(idxs,1)]
	qts=transform_ens_sol_qts(sol,t,qs)
	l=[series!(axs[i],t,qts[idxs[i],:,:]',color=:devon) for i in 1:size(idxs,1)]
	[xlims!(ax,t[1],t[end]) for ax in axs]	
	if isnothing(ylim)
		[ylims!(ax,-0.5,u0[1][2]+0.5) for ax in axs]
	elseif ylim=="auto"
		[ylims!(ax,-0.5,maximum(qts)+0.5) for (i,ax) in enumerate(axs)]
	else
		ylims!(ax,-0.5,ylim)
	end
	if !isnothing(leg)
		Legend(leg,ax)
	end
	f
end

# ╔═╡ 9870b176-cca9-4de7-9931-2b3660133c2e
function make_ensemble_plot(sol,sol_type;idxs=nothing,step=1,plot_type="qt",ylim=nothing)
	f=Figure()
	plot_ens_hist(sol["sols"][sol_type],get_ts(sol["ts"],step=step),sol["u0"];bin_width=0.2,f=f[1,1],idxs=idxs,xlim=ylim)
	if plot_type=="qt"
		plot_qt(sol["sols"][sol_type],get_ts(sol["ts"],step=step),sol["u0"];f=f[2,1],idxs=idxs,ylim=ylim)
	else
		plot_mean(sol["sols"][sol_type],get_ts(sol["ts"],step=step),sol["u0"];f=f[2,1],idxs=idxs,ylim=ylim)
	end
	f
end

# ╔═╡ 09d05769-285a-4903-abe1-c8534c061ca0
function make_single_plot(sol;idxs=nothing,ylim=nothing)
	f=Figure(size=(800,200))
	plot_sol(sol["sols"]["ode"],get_ts(sol["ts"]),sol["u0"];f=f[1,1:2],idxs=idxs,ylim=ylim)
	plot_sol(sol["sols"]["sde"],get_ts(sol["ts"]),sol["u0"];f=f[1,3:4],idxs=idxs,ylim=ylim)
	plot_sol(sol["sols"]["jump"],get_ts(sol["ts"]),sol["u0"];f=f[1,5:6],idxs=idxs,ylim=ylim,leg=f[1,7])
	f
end

# ╔═╡ 1c34bd46-c116-4ec4-9fbb-9a09681d7b7a
function plot_kld(sol,sol_type;f=Figure(),rt=0,idxs=nothing,bin_width=1,step=0.1,precompute=true)
	ax=Axis(f[1,1])
	if precompute
		kld=sol["kld"][sol_type]
	else
		kld=calculate_kld(sol,sol_type; rt=rt,idxs=idxs,bin_width=bin_width,step=step)
	end
	lines!(ax,get_ts(sol["ts"],step=step),kld)
	f
end
	

# ╔═╡ ca515e68-f9fa-4ccc-adf9-e8de8f4905d3
md"""
## All models
"""

# ╔═╡ f7232788-9741-4f5d-8790-0c79671c876a


# ╔═╡ e114fdb6-b0ef-4855-9b96-2b121fbdffd3
begin
	simple_switch=Dict()
	simple_switch["crn"] = @reaction_network begin
		@species M0(t) M1(t)
		@parameters t_on t_off l_on
		@default_noise_scaling 0.1
		@discrete_events begin
			[5] => [l ~ 1]
			# (t == t_off) => [l ~ 0.0]
		end
		(k_f1,k_b1), M0 <--> M1
		l, M0 --> M1
	end
	simple_switch["crn"]
end

# ╔═╡ 0af77422-a512-4e60-8620-eea173bdb08f
begin 
	crick=Dict()
	crick["crn"] = @reaction_network begin
		@species M0(t) M1(t) M2(t)
		@parameters t_on t_off l_on
		@default_noise_scaling 0.1
		@discrete_events begin
			((t == t_on)) => [l ~ l_on]
			((t == t_off)) => [l ~ 0.0]
		end
		(k_f1,k_b1), M0 <--> M1
		(k_f2,k_b2), M1 <--> M2
		l, M0 --> M1
	end
end

# ╔═╡ 662fbc83-03d9-4c91-a359-1c6c36365988
begin
	mts=Dict()
	mts["crn"] = @reaction_network begin
		@species M1(t) M2(t) M3(t)
		@parameters t_on t_off l_on
		@default_noise_scaling 0.2
		@discrete_events begin
			((t == t_on)) => [l ~ l_on]
			((t == t_off)) => [l ~ 0.0]
		end
		((k_f1,k_f2,k_f3),(k_b1,k_b2,k_b3)), ∅ <--> (M1, M2, M3)
		k_f2, M1 --> M1+M2
		k_f3, M2 --> M2+M3
		l,  ∅ --> M1
	end
end

# ╔═╡ 060c4219-33c9-4fe9-ba65-e412668eb31d
begin
	hillcreb(C1,C2,K_x,K_y,V,Ω) = hillar(C1/(√K_x *Ω),C2/(√K_y *Ω), V*Ω, 1, 2)
	creb=Dict()
	creb["crn"] = @reaction_network begin
		@species C1(t) C2(t)
		@parameters t_on t_off U_on U
	    @default_noise_scaling 0.2
		@discrete_events begin
	        ((t == t_on)) => [U ~ U_on]
	        ((t == t_off)) => [U ~ 0.0]
	    end
	    (Ω*r_bas_x, Ω*r_bas_y), 0 --> (C1, C2)
	    (k_dx, k_dy), (C1, C2) --> 0
	    (hillcreb(C1,C2,K_x,K_y,V_x,Ω), hillcreb(C1,C2,K_x,K_y,V_y,Ω)), 0 --> (C1,C2)
	end
end

# ╔═╡ 5b781245-497e-423e-b4a8-e02d3ea90c73
begin
	erk=Dict()
	erk["crn"] = @reaction_network begin
		@species E(t) M(t) P(t) X(t)
		@parameters t_on t_off U_on
	    @default_noise_scaling 0.2
		@discrete_events begin
	        ((t == t_on)) => [U ~ U_on]
	        ((t == t_off)) => [U ~ 0.0]
	    end
	    (U,hillar(E,0,1,K,2),M,hillar(E,0,1,K,2)), ∅ --> (E,M,P,X)
	    (γ+δ*P,γ,γ,γ), (E,M,P,X) --> ∅
	end
end

# ╔═╡ 69c0612e-30f4-445f-8c9f-8589d5313502
begin
	pkm=Dict()
	pkm["crn"] = @reaction_network begin
		@species PKM(t) FActin(t) RNA(t) PKMa(t) FActina(t) RNAa(t)
		@parameters t_on t_off U_on
	    
		@default_noise_scaling 0.2
	    @discrete_events begin
	        ((t == t_on)) => [U ~ U_on]
	        ((t == t_off)) => [U ~ 0.0]
	    end
	    (j_1*RNA/τ_1,1/τ_1), PKM <--> PKMa
	    (j_2+j_3*PKMa/τ_2,1/τ_2), FActin <--> FActina
	    (j_4*FActina*(PKMa+U)/τ_3,1/τ_3) ,RNA <--> RNAa
	end
end

# ╔═╡ ce3407d1-c524-43a6-830d-fbba078bbc75
md"""
Epi equations 
"""


# %   1. D                   --    kkw1         --> Dm1
# %   2. D                   --    kkm1         --> Dm1
# %   3. Dm1                 --    kke1         --> D
# %   4. Dm1                 --    delta1       --> D
# %   5. Dm1                 --    kkke1        --> D
# %   6. D                   --    kkw1         --> Dm2
# %   7. D                   --    kkm1         --> Dm2
# %   8. Dm2                 --    kke1         --> D
# %   9. Dm2                 --    delta1       --> D
# %  10. Dm2                 --    kkke1        --> D

# ======================================================

# %   1. D                  --    kw10           --> D1
# %   2. D                  --    kw1            --> D1
# %   3. D                  --    kmprime        --> D1
# %   4. D                  --    kmprime        --> D1
# %   5. D1                 --    deltaprime     --> D
# %   6. D1                 --    ktprime        --> D 
# %   7. D1                 --    ktprimeact     --> D
# %   8. D1                 --    kw20           --> D12
# %   9. D1                 --    km             --> D12
# %  10. D1                 --    km             --> D12
# %  11. D1                 --    kmbar          --> D12
# %  12. D1                 --    kmbar          --> D12
# %  13. D12                --    delta          --> D1
# %  14. D12                --    ke             --> D1
# %  15. D12                --    keact          --> D1
# %  16. D                  --    kw20           --> D2
# %  17. D                  --    kw2            --> D2
# %  18. D                  --    km             --> D2
# %  19. D                  --    km             --> D2
# %  20. D                  --    kmbar          --> D2
# %  21. D                  --    kmbar          --> D2
# %  22. D2                 --    delta          --> D
# %  23. D2                 --    ke             --> D
# %  24. D2                 --    keact          --> D
# %  25. D2                 --    kw10           --> D12
# %  26. D2                 --    kmprime        --> D12
# %  27. D2                 --    kmprime        --> D12
# %  28. D12                --    deltaprime     --> D2
# %  29. D12                --    ktprime        --> D2
# %  30. D12                --    ktprimeact     --> D2
# %  31. D                  --    kwa0           --> Da
# %  32. D                  --    kwa            --> Da
# %  33. D                  --    kma            --> Da
# %  34. Da                 --    delta          --> D
# %  35. Da                 --    kea            --> D
# %  36. Da                 --    keacta         --> D
# %  37. Da                 --    keacta         --> D
# %  38. Da                 --    keacta         --> D
# %  39. Da                 --    keacta         --> D

# ╔═╡ 6f7c6b6f-195e-4283-905f-b956bed191ab
epi_simple = @reaction_network begin
	(kkw1+kkm1,kke1+delta1+kkke1), D <--> Dm1
	(kkw1+kkm1,kke1+delta1+kkke1), D <--> Dm2
end

# ╔═╡ a382e5b6-abd4-48b1-bfed-be2793e20bb0
epi = @reaction_network begin
	(kw1010+kw1+kmprime+kmprime,deltaprime+ktprime+ktprimeact), D <--> D1
	(kw20+km+km+kmbar+kmbar,delta+ke+keact), D1 <--> D12
	(kw20+kw2+km+km+kmbar+kmbar,delta+ke+keact), D <--> D2
	(kw10+kmprime+kmprime,deltaprime+ktprime+ktprimeact), D2 <--> D12
	(kwa0+kwa+kma, delta+kea+keacta+keacta+keacta+keacta), D <--> Da
end

# ╔═╡ 75527d17-bb0d-46dd-a462-6ab166b5bd82
md"""
## Simple switch
"""

# ╔═╡ cd1175f0-136b-46de-92d7-cfbd2ecc4987
simple_switch["crn"]

# ╔═╡ ac305374-c773-4359-b2b5-459112c65460
latexify(simple_switch["crn"]; form = :ode)

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

# ╔═╡ 5d15ba23-3744-40dd-b350-7a2eddfac4ad
begin
	simple_switch["ts"]=(0.,300.)
	simple_switch["u0"]=[:M0 => 100, :M1 => 0]
	simple_switch["ps"]=[:l=>0,:k_f1 => 0.005,:k_b1 => 0.1,:t_on=>0,:t_off=>0,:l_on=>0,:l=>0]
	merge!(simple_switch,make_models(simple_switch))
end

# ╔═╡ 4cc91eba-cf05-4460-8b36-f448c4f59099


# ╔═╡ 5798599e-e273-4029-82ff-6ac7aa27bcc1
begin
	model=simple_switch
	ode=ODEProblem(model["crn"],model["u0"],model["ts"],model["ps"])
	# jinput = JumpInputs(model["crn"], u_int(simple_switch["u0"]), model["ts"],model["ps"])
	# jprob = JumpProblem(jinput)
	# eprob_jump=EnsembleProblem(jprob)
end

# ╔═╡ 6cc6d06b-94f7-4492-9a07-48ac3595a33c
s=solve(ode)

# ╔═╡ c2201c75-24fe-487c-a51b-7760014248a7
plot_sol(s,model["u0"],model["ts"])

# ╔═╡ 031f79f3-340f-4424-b124-1f1f000cc960
sde=SDEProblem(model["crn"],model["u0"],model["ts"],model["ps"])

# ╔═╡ 95a1aa00-58fe-4202-a2bd-cfedd64691c9
solve(sde)

# ╔═╡ 82529b97-0e9d-419c-b30f-a1f405bf6bb6
ens=EnsembleProblem(sde)

# ╔═╡ e321d9e9-8dd4-49ee-bf42-dd67467c5958
x=solve(ens;trajectories=1000)

# ╔═╡ 0c164361-5264-4baa-b7ba-c7cb398f2b96
plot_ens_hist(x,model["ts"],model["u0"])

# ╔═╡ 5da94219-164d-476f-933a-2f37e7d4feae
jinput

# ╔═╡ 0fe8b08c-9ac9-4f58-ae36-caa9affd95e5
JumpProblem(jinput)

# ╔═╡ 3de8ca3a-793b-4faf-bbd8-92a0d75898ef
make_models(simple_switch)

# ╔═╡ bc97d259-e6a7-42d8-80f7-f6e29e3cdfa0
begin
	simple_switch["ts"]=(0.,300.)
	simple_switch["u0"]=[:M0 => 1, :M1 => 0]
	simple_switch["ps"]=vcat([:k_f1 => 0.005,:k_b1 => 0.1],stim)
	merge!(simple_switch,make_models(simple_switch))
end

# ╔═╡ eaa47f4a-9d83-4f73-9076-363ea84f3f45
simple_switch["sols"]=solve_all(simple_switch;t_on=120,t_off=170,trajectories=100);

# ╔═╡ 5757f358-12bf-4040-8437-fa0171b5f032
make_single_plot(simple_switch)

# ╔═╡ fae1bed4-d76b-4c4b-8456-499feac5d225
make_ensemble_plot(simple_switch,"sde_ens";step=1)

# ╔═╡ e1d6348c-8c0b-421b-987d-7b5f1afd91fd
make_ensemble_plot(simple_switch,"jump_ens";step=1)

# ╔═╡ 7a633cae-4afc-448e-83e2-1201a32670a4
simple_switch["kld"]=Dict()

# ╔═╡ 7b14cc10-f393-47e5-b0a7-9c537c7e1b86
simple_switch["kld"]["jump_ens"]=calculate_kld(simple_switch,"jump_ens";step=1)

# ╔═╡ cb1ed683-e84e-439e-8fae-25452cc291b2
plot_kld(simple_switch,"jump_ens";step=1)

# ╔═╡ ba7b52cc-0173-4342-8b44-1e7bbe28bb9e
PlutoUI.ExperimentalLayout.grid([
	md"# Layout demo!"      Text("")
	Text("I'm on the left") Dict(:a => 1, :b => [2,3])
])

# ╔═╡ dcb5cb12-8e79-48a6-9320-1348fb92a330
md"""
### Multiple switches
"""

# ╔═╡ 0b3c11d0-6f48-499c-9b57-72c0ea2edf40
begin
	multi_switch=Dict()
	multi_switch["crn"]=simple_switch["crn"]
	multi_switch["ts"]=simple_switch["ts"]
	multi_switch["ps"]=simple_switch["ps"]
end

# ╔═╡ 9f9d5a5e-7b44-470a-8d9f-7e3182afed9d
begin
	multi_switch["u0"]=[:M0 => 100, :M1 => 0]
	merge!(multi_switch,make_models(multi_switch))
end

# ╔═╡ 8f23cd47-8cc4-4c5b-ba64-afc45ff274d8
multi_switch["sols"]=solve_all(multi_switch;t_on=120,t_off=170,trajectories=100);

# ╔═╡ 755ef828-e5dd-4b83-9272-bd396c898d4d
make_single_plot(multi_switch)

# ╔═╡ a03d42f1-fe47-4366-ad74-c0dcaed958a5
make_ensemble_plot(multi_switch,"sde_ens";step=1)

# ╔═╡ e48fa896-6e40-4bcf-9d80-b97dca7b1951
make_ensemble_plot(multi_switch,"jump_ens";step=1)

# ╔═╡ d198d116-e803-4f3f-9814-4fe5bc4b5bd8
multi_switch["kld"]=Dict()

# ╔═╡ 6087db7a-9190-4133-9bd9-706ed44ff00a


# ╔═╡ d3184722-630a-4d81-a186-7b747ffc9914
md"""
## Crick switches
"""

# ╔═╡ 3ed7b1d6-d4be-4500-8a51-af57bc2641a9
crick["crn"]

# ╔═╡ 2364af6c-d6d6-4045-9441-c494ccb66981
begin
	crick["ts"]=(0.,300.)
	crick["u0"]=[:M0 => 1, :M1 => 0, :M2 => 0]
	crick["ps"]=vcat([:k_f1 => 0.005,:k_b1 => 0.1,:k_f2 => 0.05,:k_b2 => 0.1],stim)
	merge!(crick,make_models(crick))
end

# ╔═╡ 2f381ec5-2fd3-4a87-8dfd-5239bf9ed8b5
crick["sols"]=solve_all(crick;t_on=120,t_off=170,trajectories=100);

# ╔═╡ edf776ba-23db-4240-9a69-4c2d738f7e07
make_single_plot(crick)

# ╔═╡ 0e6119f4-0f1f-49dd-8e87-ca26b86d3f68
make_ensemble_plot(crick,"sde_ens";step=1)

# ╔═╡ ee4b9acd-305d-41c4-928b-316e149942b8
make_ensemble_plot(crick,"jump_ens";step=1)

# ╔═╡ 89814c74-6a4d-4f57-a384-dd07697a39fc
crick["kld"]=Dict()

# ╔═╡ 24ef355a-0e28-4512-b7df-b7214c38070f
crick["kld"]["jump_ens"]=calculate_kld(crick,"jump_ens";step=1,xlim="auto")

# ╔═╡ 6fbf7de8-bd5f-4c43-9580-a7d47c613003
md"""
Compare simple switch and Crick switch
"""

# ╔═╡ f53958f8-dd41-41f1-bc4c-134339998a24
plot_kld(hcat(simple_switch["kld"],crick["kld"])',crick["ts"],step=1)

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
Here we have a four-stage switch where each species triggers the production of the following species rather than being converted to the next species. Every species should get direct input from the stimulus as a control. From a mass-action point of view, the main difference between one species governing the rate of production of the next species (this model) vs. getting converted to the next species (Crick switch) is that the quantity of the first species does not decrease as the subsequent species gets produced, and therefore the timescale of each level is longer.
"""

# ╔═╡ 988edb53-2f7a-4aad-9720-8d4a45f32adc
mts["crn"]

# ╔═╡ 17aacbe0-0f5e-4115-8407-e0547832c18f
begin
	mts["ts"]=(0.,1000.)
	mts["u0"]=[:M1 => 0,:M2 => 0, :M3 =>0]
	mts["ps"]=vcat([[[Symbol("k_f$i") => 0.01*(0.2^i), Symbol("k_b$(i)") => 0.2^i] for i in 1:3]..., stim]...)
	merge!(mts,make_models(mts))
end

# ╔═╡ 95d4e4c8-ac84-4caa-b961-f41b6fc52d8f
@time mts["sols"]=solve_all(mts;t_on=120,t_off=170,trajectories=100);

# ╔═╡ b9dd6e51-9e95-4aae-93d4-b23b2982a763
make_single_plot(mts;ylim="auto")

# ╔═╡ 817e4d6f-561a-44e8-bf56-eeb8f40ddf2e
make_ensemble_plot(sol_mts["sde_ens"],ts_mts,u0_mts;idxs=[2,4,6,8],step=1)

# ╔═╡ 5b8c14ad-2aeb-405b-9f55-bfa0783fa8ec
make_ensemble_plot(sol_mts["jump_ens"],ts_mts,u0_mts;idxs=[2,4,6,8],step=1)

# ╔═╡ 8217284d-0f43-40e6-a434-aa3d0ad45b30
make_ensemble_plot(sol_mts["jump_ens"],ts_mts,u0_mts;idxs=[2,4,6,8],plot_type="mean",step=1)

# ╔═╡ 46c2e1c8-565b-4f2a-8b42-ff34b0b83c90
md"""
## CREB
"""

# ╔═╡ c2a5f9f9-fbbb-4f64-807b-611a6c67e3cb
creb["crn"]

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

# ╔═╡ Cell order:
# ╟─6518c587-8638-4909-a995-b331634c8079
# ╠═9cbb14a7-884b-4783-8fb4-650260d940d4
# ╟─524af17b-ef72-4a03-a471-585db57f3ec1
# ╠═80447764-5654-4235-87b5-f447e68a8438
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
# ╠═e114fdb6-b0ef-4855-9b96-2b121fbdffd3
# ╠═0af77422-a512-4e60-8620-eea173bdb08f
# ╠═662fbc83-03d9-4c91-a359-1c6c36365988
# ╠═060c4219-33c9-4fe9-ba65-e412668eb31d
# ╠═5b781245-497e-423e-b4a8-e02d3ea90c73
# ╠═69c0612e-30f4-445f-8c9f-8589d5313502
# ╠═ce3407d1-c524-43a6-830d-fbba078bbc75
# ╠═6f7c6b6f-195e-4283-905f-b956bed191ab
# ╠═a382e5b6-abd4-48b1-bfed-be2793e20bb0
# ╠═75527d17-bb0d-46dd-a462-6ab166b5bd82
# ╠═cd1175f0-136b-46de-92d7-cfbd2ecc4987
# ╠═ac305374-c773-4359-b2b5-459112c65460
# ╠═7f8a93fc-07f9-4ec5-918d-ff4dc7f40f27
# ╟─f5cc3e6d-73cf-4c23-a075-31008153db3b
# ╠═5d15ba23-3744-40dd-b350-7a2eddfac4ad
# ╠═4cc91eba-cf05-4460-8b36-f448c4f59099
# ╠═5798599e-e273-4029-82ff-6ac7aa27bcc1
# ╠═6cc6d06b-94f7-4492-9a07-48ac3595a33c
# ╠═c2201c75-24fe-487c-a51b-7760014248a7
# ╠═031f79f3-340f-4424-b124-1f1f000cc960
# ╠═95a1aa00-58fe-4202-a2bd-cfedd64691c9
# ╠═82529b97-0e9d-419c-b30f-a1f405bf6bb6
# ╠═e321d9e9-8dd4-49ee-bf42-dd67467c5958
# ╠═0c164361-5264-4baa-b7ba-c7cb398f2b96
# ╠═5da94219-164d-476f-933a-2f37e7d4feae
# ╠═0fe8b08c-9ac9-4f58-ae36-caa9affd95e5
# ╠═3de8ca3a-793b-4faf-bbd8-92a0d75898ef
# ╠═bc97d259-e6a7-42d8-80f7-f6e29e3cdfa0
# ╠═eaa47f4a-9d83-4f73-9076-363ea84f3f45
# ╠═5757f358-12bf-4040-8437-fa0171b5f032
# ╠═fae1bed4-d76b-4c4b-8456-499feac5d225
# ╠═e1d6348c-8c0b-421b-987d-7b5f1afd91fd
# ╠═7a633cae-4afc-448e-83e2-1201a32670a4
# ╠═7b14cc10-f393-47e5-b0a7-9c537c7e1b86
# ╠═cb1ed683-e84e-439e-8fae-25452cc291b2
# ╠═ba7b52cc-0173-4342-8b44-1e7bbe28bb9e
# ╠═dcb5cb12-8e79-48a6-9320-1348fb92a330
# ╠═0b3c11d0-6f48-499c-9b57-72c0ea2edf40
# ╠═9f9d5a5e-7b44-470a-8d9f-7e3182afed9d
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
