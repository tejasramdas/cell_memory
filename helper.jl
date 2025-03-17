### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 5aa9cdd3-d1eb-4b5a-a007-6ad52b7b7c88
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

# ╔═╡ 8f2eaf49-5229-4994-99fd-f2de6c2844e7
md"""
## Helper functions
"""


# ╔═╡ 04697b3d-60df-4c03-a605-693f52d52045
u_int(u) = map(i->Pair(i[1],Int(i[2])),u)

# ╔═╡ 3b0704f4-d738-467f-8899-91040d29254a
function get_ts(ts;step=0.1)
	ts[1]:step:ts[2]
end

# ╔═╡ a9cccc21-66c4-46b5-965b-95398afb3199
function transform_ens_sol(sol,t,u0)
	concat_sol=hcat(timeseries_point_mean(sol,t)...)
end
export transorm_ens_sol

# ╔═╡ b915053d-8a03-4729-a415-4b5b58c75834
function transform_sol(sol,t;idxs=nothing)
	hcat(sol.(t,idxs=idxs)...)
end

# ╔═╡ a165a3f0-440f-4d65-8985-a8c07283819a
get_labels(u) = map(i->i[1],u)

# ╔═╡ 1ef44785-401c-423b-bca1-001fde90a684
function make_models(model,u0,tspan,ps;u0_int=nothing,jump_p=true)
    ode=ODEProblem(model,u0,tspan,ps)
    sde=SDEProblem(model, u0, tspan, ps)
    if jump_p
		if isnothing(u0_int)
			u0_int=u_int(u0)
		end
        jinput = JumpInputs(model, u0_int, tspan, ps)
        jprob = JumpProblem(jinput)
        eprob_jump=EnsembleProblem(jprob)
    else
        jprob=nothing
        eprob=nothing
    end
	eprob_sde = EnsembleProblem(sde)
    return Dict(["ode" => ode,"sde"=> sde, "jump" => jprob, "eprob_sde" => eprob_sde, "eprob_jump" => eprob_jump])
end

# ╔═╡ 8f8d5ed5-3b58-423c-a5e2-101f16ced163
function solve_all(models;t_on=0,t_off=0,trajectories=1000,ensemble=true)
	tstops=[t_on,t_off]
	sols=map(i->(i=>solve(models[i];tstops=tstops)),["ode","sde","jump"])
	if ensemble
		push!(sols,("sde_ens" => solve(models["eprob_sde"],STrapezoid();tstops=tstops,trajectories=trajectories)))
		push!(sols,("jump_ens" => solve(models["eprob_jump"],SSAStepper();tstops=tstops,trajectories=trajectories)))
	end
	Dict(sols)
end

# ╔═╡ bfd7af99-f571-41bd-bf1b-077ecd33d35c
function calculate_kld(sol, ts, u0; rt=0,idxs=nothing,bin_width=1)
	if isnothing(idxs)
		idxs=1:size(u0,1)
		i_out=idxs[end]
	end
	kl_divs=[]
	bins=Tuple([-bin_width:bin_width:u0[idx][2]+bin_width for idx in idxs])
	h_rt=fit(Histogram, Tuple(componentwise_vectors_timepoint(sol,rt)[idxs]),bins)
	for t in ts
		h=fit(Histogram, Tuple(componentwise_vectors_timepoint(sol,t)[idxs]),bins)
		push!(kl_divs,kldivergence(h_rt.weights,h.weights))
	end
	kl_divs
end

# ╔═╡ a77b2a37-71ec-4a05-af72-970e44a4817e
function plot_sol(sol,t,u0;f=Figure(),idxs=nothing,leg=nothing,ylim=nothing)
	if isnothing(idxs)
		idxs=1:size(u0,1)
	end
	ax=Axis(f[1,1])
	plot_sol=transform_sol(sol,t,idxs=idxs)
	l=series!(ax,t,plot_sol,labels=["$(u0[idxs[i]][1])" for i in 1:size(idxs,1)])
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


# ╔═╡ 5753f247-065a-4a1c-8c95-d3eb2b8e4a80
function plot_kld(kld,ts;f=Figure())
	ax=Axis(f[1,1])
	lines!(ax,ts,kld)
	f
end

# ╔═╡ cfa3843f-2ade-43fd-9cfb-8d92d26a5b44
function plot_ens_hist(sol,t,u0;bin_width=1,f=Figure(),leg=nothing,idxs=nothing)
	if isnothing(idxs)
		idxs=1:size(u0,1)
	end
	ax=Axis(f[1,1])
	s=Makie.Slider(f[2,1],range = t)
	l=[hist!(ax,@lift(componentwise_vectors_timepoint(sol,$(s.value))[idxs[i]]),label="$(get_labels(u0)[idxs[i]])",bins=(-bin_width:bin_width:u0[1][2]+bin_width)) for i in 1:size(idxs,1)]
	if !isnothing(leg)
		Legend(leg,ax)
	end
	xlims!(ax,-bin_width,u0[1][2]+bin_width)
	title_lab = Label(f[0,1],@lift("Timepoint $($(s.value))"),tellwidth=false)
	f
end

# ╔═╡ 8c129f6f-90e8-4872-b87e-0478411bbf8f
function plot_qt(sol,t,u0;qs=0.25:0.25:1,idxs=nothing,f=Figure())
	if isnothing(idxs)
		idxs=1:size(u0,1)
	end
	axs=[Axis(f[1:5,i]) for i in 1:size(idxs,1)]
	println("HHELO")
	@time qts=[(q => timeseries_point_quantile(sol,q,t)) for q in qs]
	@time l=[[lines!(axs[i],t,qt[2][idxs[i],:],alpha=0.8-(abs(0.5-qt[1])/2.5),color=:blue) for qt in qts] for i in 1:size(idxs,1)]
	[xlims!(ax,t[1],t[end]) for ax in axs]
	[ylims!(ax,-0.5,u0[1][2]+0.5) for ax in axs]
	f
end

# ╔═╡ 75fc85e7-6026-40dd-8d22-5e672faa9000
function make_ensemble_plot(sol, ts,u0;idxs=nothing)
	f=Figure()
	plot_ens_hist(sol,get_ts(ts),u0;bin_width=0.2,f=f[1,1],idxs=idxs)
	plot_qt(sol,get_ts(ts),u0;f=f[2,1],idxs=idxs)
	f
end

# ╔═╡ 2841fa26-6979-49d7-82ab-beba39719c87
function make_single_plot(sol,ts,u0;idxs=nothing,ylim=nothing)
	f=Figure(size=(800,200))
	plot_sol(sol["ode"],get_ts(ts),u0;f=f[1,1:2],idxs=idxs,ylim=ylim)
	plot_sol(sol["sde"],get_ts(ts),u0;f=f[1,3:4],idxs=idxs,ylim=ylim)
	plot_sol(sol["jump"],get_ts(ts),u0;f=f[1,5:6],idxs=idxs,ylim=ylim,leg=f[1,7])
	f
end

# ╔═╡ b4c321bc-cc76-4b73-8f84-b20e1aef0206
function testrevise()
	println("Hello")
end

# ╔═╡ Cell order:
# ╠═5aa9cdd3-d1eb-4b5a-a007-6ad52b7b7c88
# ╠═8f2eaf49-5229-4994-99fd-f2de6c2844e7
# ╠═04697b3d-60df-4c03-a605-693f52d52045
# ╠═3b0704f4-d738-467f-8899-91040d29254a
# ╠═a9cccc21-66c4-46b5-965b-95398afb3199
# ╠═b915053d-8a03-4729-a415-4b5b58c75834
# ╠═a165a3f0-440f-4d65-8985-a8c07283819a
# ╠═1ef44785-401c-423b-bca1-001fde90a684
# ╠═8f8d5ed5-3b58-423c-a5e2-101f16ced163
# ╠═bfd7af99-f571-41bd-bf1b-077ecd33d35c
# ╠═a77b2a37-71ec-4a05-af72-970e44a4817e
# ╠═5753f247-065a-4a1c-8c95-d3eb2b8e4a80
# ╠═cfa3843f-2ade-43fd-9cfb-8d92d26a5b44
# ╠═8c129f6f-90e8-4872-b87e-0478411bbf8f
# ╠═75fc85e7-6026-40dd-8d22-5e672faa9000
# ╠═2841fa26-6979-49d7-82ab-beba39719c87
# ╠═b4c321bc-cc76-4b73-8f84-b20e1aef0206
