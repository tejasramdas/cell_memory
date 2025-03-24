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
  using LinearAlgebra
  using ProgressBars
end


u_int(u) = map(i->Pair(i[1],Int(i[2])),u)
function get_ts(ts;step=0.1)
	ts[1]:step:ts[2]
end
function transform_sol(sol,t;idxs=nothing)
	hcat(sol.(t,idxs=idxs)...)
end
function transform_ens_sol_qts(sol,t,qts)
	concat_sol=cat([hcat(timeseries_point_quantile(sol,q,t)...) for q in qts]...,dims=3)
end
function transform_ens_sol(sol,t)
	concat_sol=cat([cat(componentwise_vectors_timepoint(sol,i)...,dims=2) for i in t]...,dims=3)
	permutedims(concat_sol,[2,3,1])
end
function transform_ens_sol_mean(sol,t)
	concat_sol_qts=hcat(timeseries_point_mean(sol,t)...)
end
get_labels(u) = map(i->i[1],u)
function make_models(model)
    ode=ODEProblem(model["crn"],model["u0"],model["ts"],model["ps"])
    sde=SDEProblem(model["crn"],model["u0"],model["ts"],model["ps"])
    u0_int=u_int(model["u0"])
	jinput = JumpInputs(model["crn"], u0_int, model["ts"],model["ps"])
	jprob = JumpProblem(jinput)
	eprob_jump=EnsembleProblem(jprob)
    eprob_sde = EnsembleProblem(sde)
    return Dict(["ode" => ode,"sde"=> sde, "jump" => jprob, "eprob_sde" => eprob_sde, "eprob_jump" => eprob_jump])
end
function solve_all(models;trajectories=1000,ensemble=true)
	ps=Dict(models["ps"])
	tstops=[ps[:t_on],ps[:t_off]]
  sols=map(i->(i=>solve(models["models"][i];tstops=tstops)),["ode","sde","jump"])
	if ensemble
      push!(sols,("sde_ens" => solve(models["models"]["eprob_sde"];tstops=tstops,trajectories=trajectories)))
      push!(sols,("jump_ens" => solve(models["models"]["eprob_jump"],SSAStepper();tstops=tstops,trajectories=trajectories)))
	end
	Dict(sols)
end
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
      xlim=maximum([maximum(hcat(sol[i].u...)[idxs,:]) for i in 1:size(sol,3)])
	else
		xlim=xlim
	end
  bin_width=max(bin_width,Int(floor(xlim/100)))
	bins=Tuple([-bin_width/2:bin_width:xlim+bin_width/2 for idx in idxs])
  h_rt=fit(Histogram, Tuple(componentwise_vectors_timepoint(sol,rt)[idxs]),bins)
  for t in ProgressBar(get_ts(ts,step=step))
      h=fit(Histogram, Tuple(componentwise_vectors_timepoint(sol,t)[idxs]),bins)
      push!(kl_divs,kldivergence(h.weights/sum(h.weights) .+ 1e-6,h_rt.weights/sum(h_rt.weights) .+ 1e-6))
	end
	kl_divs
end
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
function make_ensemble_plot(sol,sol_type;f=Figure(size=(800,600)),idxs=nothing,bin_width=1,step=1,plot_type="qt",ylim=nothing)
	plot_ens_hist(sol["sols"][sol_type],get_ts(sol["ts"],step=step),sol["u0"];bin_width=bin_width,f=f[1,1],idxs=idxs,xlim=ylim)
	if plot_type=="qt"
		plot_qt(sol["sols"][sol_type],get_ts(sol["ts"],step=step),sol["u0"];f=f[2,1],idxs=idxs,ylim=ylim)
	else
		plot_mean(sol["sols"][sol_type],get_ts(sol["ts"],step=step),sol["u0"];f=f[2,1],idxs=idxs,ylim=ylim)
	end
	f
end
function make_single_plot(sol;f=Figure(size=(1500,500)),idxs=nothing,ylim=nothing)
	plot_sol(sol["sols"]["ode"],get_ts(sol["ts"]),sol["u0"];f=f[1,1:2],idxs=idxs,ylim=ylim)
	plot_sol(sol["sols"]["sde"],get_ts(sol["ts"]),sol["u0"];f=f[1,3:4],idxs=idxs,ylim=ylim)
	plot_sol(sol["sols"]["jump"],get_ts(sol["ts"]),sol["u0"];f=f[1,5:6],idxs=idxs,ylim=ylim,leg=f[1,7])
	f
end
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
function remake_models(old_model;u0=Dict())
    model=Dict()
    model["crn"]=deepcopy(old_model["crn"])
    model["ts"]=old_model["ts"]
    model["ps"]=copy(old_model["ps"])
    model["models"]=deepcopy(old_model["models"])
    for (k,m) in model["models"]
        model["models"][k]=remake(m,u0=u0)
    end
    for i in ["sols","plots","kld"] 
        try
            delete!(model,i)
        catch
        end
    end
    model["u0"]=u0
    model
end
function compute_all!(model;trajectories=100,kld=true,ensemble=true)
    model["sols"]=solve_all(model;trajectories=trajectories,ensemble=ensemble)
    @info "Solutions computed"
    model["plots"]=Dict()
    model["plots"]["single"]=make_single_plot(model,ylim="auto")
    if ensemble
        model["plots"]["jump_ens"]=make_ensemble_plot(model,"jump_ens";step=0.1,ylim="auto")
    end
    @info "Plots made"
    if kld
        model["kld"]=Dict()
        model["kld"]["jump_ens"]=calculate_kld(model,"jump_ens";rt=t_on-10,step=1,xlim="auto")
        model["plots"]["jump_kld"]=plot_kld(model,"jump_ens";bin_width=1,step=1)
        @info "KLD computed"
    end
end
