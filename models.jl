# Fetch required packages.
using Catalyst, OrdinaryDiffEq, Debugger
using JumpProcesses
using StochasticDiffEq, DifferentialEquations
using GLMakie
using DifferentialEquations.EnsembleAnalysis

test = @reaction_network begin
    (k_f, k_b), M0 <--> M1
end

simple_binary = @reaction_network begin
    @default_noise_scaling 0.2
    @discrete_events begin
        ((t == 20)) => [l ~ 1.0]
        ((t == 70)) => [l ~ 0.0]
    end
    (k_f*(10*l+1), k_b), M0 <--> M1
end

crick = @reaction_network begin
    @default_noise_scaling 0.2
    @discrete_events begin
        ((t == 20)) => [l ~ 1.0]
        ((t == 70)) => [l ~ 0.0]
    end
    (k_f1*(10*l+1),k_b1), M0 <--> M1
    (k_f2,k_b2), M1 <--> M2
end

tspan=(0.,200.)
tstops=[20,70]
u_int(u) = map(i->Pair(i[1],Int(i[2])),u)
function solve_plot(model,name,prob,tspan,tstops,ax;plot_symbs=[],axj=nothing,leg=false)
    empty!(ax)    
    sol=solve(prob;tstops)
    vals=hcat(sol.(tspan[1]:0.1:tspan[2];idxs=plot_symbs)...)
    s=series!(ax,tspan[1]:0.1:tspan[2],vals,labels=map(u->"$(u)",plot_symbs))
    if axj != nothing
        lines!(axj,tspan[1]:0.1:tspan[2],vals'[:,end],label="$name")
        if name=="Crick"
            axislegend(axj)
        end
    end
    axislegend(ax)
    return vals[end,:],s
end

function ens_solve_plot(model,name,prob,tspan,tstops,ax;axj=nothing,leg=false)
    empty!(ax)    
    sol=solve(prob,SSAStepper();tstops,trajectories=1000)
    solsum=sq(sol,tspan)
    for i in [[0.05,0.95],[0.1,0.9],[0.25,0.75]]
        band!(ax,tspan[1]:0.1:tspan[2],solsum[i[1]][1,:],solsum[i[2]][1,:],color=:blue,alpha=0.3)#,labels=map(u->"$(u[1])",u0))
        band!(ax,tspan[1]:0.1:tspan[2],solsum[i[1]][2,:],solsum[i[2]][2,:],color=:aqua,alpha=0.3)#,labels=map(u->"$(u[1])",u0))
    end
    lines!(ax,tspan[1]:0.1:tspan[2],solsum[0.5][1,:],color=:blue)#,labels=map(u->"$(u[1])",u0))
    lines!(ax,tspan[1]:0.1:tspan[2],solsum[0.5][2,:])#,labels=map(u->"$(u[1])",u0))
    lines!(axj,tspan[1]:0.1:tspan[2],solsum[0.5][2,:])#,labels=map(u->"$(u[1])",u0))
    # vals=hcat(sol.()...)
    # if axj != nothing
        # lines!(axj,tspan[1]:0.1:tspan[2],vals'[:,end],label="$name")
        # if name=="Crick"
            # axislegend(axj)
        # end
    # end
    # axislegend(ax)
    # return vals[end,:]
end

function make_models(model,u0,tspan,ps;jump_p=true)
    ode=ODEProblem(model,u0,tspan,ps)
    sde=SDEProblem(model, u0, tspan, ps)
    if jump_p
        jinput = JumpInputs(model, u_int(u0), tspan, ps)
        jprob = JumpProblem(jinput)
        eprob=EnsembleProblem(jprob)
    else
        jprob=nothing
        eprob=nothing
    end
    return ode,sde,jprob,eprob
end


function sq(esol,tspan)
    sols=[]
    for i in [0.05,0.1,0.25,0.5,0.75,0.9,0.95]
        push!(sols,i => timeseries_point_quantile(esol, i, tspan[1]:0.1:tspan[2]))
    end
    return Dict(sols)
end

function solve_all(model,name,u0,ps,tspan,tstops,axs,axsj)
    probs = make_models(model,u0,tspan,ps)
    for (i,j) in enumerate(probs[1:end-2])
        solve_plot(model,name,j,tspan,tstops,axs[i];axj=axsj[i])
    end
    ens_solve_plot(model,name,probs[4],tspan,tstops,axs[3];axj=axsj[3])
end

crick_figure=Figure(size=(800,800))
crick_ax=hcat([vcat([Axis(crick_figure[i,j],limits=(0,200,0,10)) for i in 1:3]...) for j in 1:3]...)
for i in 1:3
    crick_ax[1,i].title = ["Binary","Crick","Overlaid"][i]
end
for i in 1:3
    ls = Label(crick_figure[i,0],["ODE","SDE","Jump"][i],rotation=π/2,tellheight=false)
end
display(crick_figure);


model=simple_binary
u0 = [:M0 => 10.0,:M1 => 0.0]
ps=[:k_f => 0.005,:k_b=>0.1, :l=>0.0]
solve_all(model,"Binary",u0,ps,tspan,tstops,crick_ax[:,1],crick_ax[:,3])


model=crick

u0 = [:M0 => 10.0,:M1 => 0.0,:M2 => 0.0]
ps=[:k_f1 => 0.005,:k_b1=>0.1,:k_f2 => 0.5,:k_b2=>0.1,:l=>0.0]
solve_all(model,"Crick",u0,ps,tspan,tstops,crick_ax[:,2],crick_ax[:,3])


# hillar(X,Y,v,K,n) = v*(X^n) / (X^n + Y^n + K^n)
hillcreb(C1,C2,K_x,K_y,V,Ω) = hillar(C1/(√K_x *Ω),C2/(√K_y *Ω), V*Ω, 1, 2)
creb = @reaction_network begin
    (Ω*r_bas_x, Ω*r_bas_y), 0 --> (C1, C2)
    (k_dx, k_dy), (C1, C2) --> 0
    (hillcreb(C1,C2,K_x,K_y,V_x,Ω), hillcreb(C1,C2,K_x,K_y,V_y,Ω)), 0 --> (C1,C2)
end

tspan=(0.,500.)
tstops=[20,70]
vals=[]
for i in 1:5
    ps[1]=(:V_x => 0.2*(1+i))
    ode=ODEProblem(creb,u0,tspan,ps)
    sol=solve(ode)
    push!(vals,hcat(sol.(tspan[1]:0.1:tspan[2])...))
end

g=Figure()
ax=[]
for i in 1:5
    push!(ax,Axis(g[1,i]))
    series!(ax[i],tspan[1]:0.1:tspan[2],vals[i],labels=map(u->"$(u[1])",u0))
end
axislegend(ax[1])

# u0 = [:C1 => 10.0, :C2 => 0]
# u0_integers = [:C1 => 10, :C2 => 0]
# tspan = (0., 1000.)
# ps= [:Ω => 10, :V_x => 0.24, :V_y => 0.01, :K_x => 5, :K_y => 10, :k_d_x => 0.04, :k_d_y => 0.01, :r_bas_x => 0.003, :r_bas_y => 0.002]
#

erk = @reaction_network begin
    @discrete_events begin
        ((t == 20)) => [U ~ 10.0]
        ((t == 70)) => [U ~ 0.0]
    end
    (U,hillar(E,0,1,K,2),M,hillar(E,0,1,K,2)), 0 --> (E,M,P,X)
    (γ+δ*P,γ,γ,γ), (E,M,P,X) --> 0
end

pkm = @reaction_network begin
    @discrete_events begin
        ((t == 20)) => [U ~ 10.0]
        ((t == 70)) => [U ~ 0.0]
    end
    (j_1*RNA/τ_1,1/τ_1), PKM <--> PKMa
    (j_2+j_3*PKMa/τ_2,1/τ_2), FActin <--> FActina
    (j_4*FActina*(PKMa+U)/τ_3,1/τ_3) ,RNA <--> RNAa
end



f=Figure()
ax=[]
tits=["CREB","ERK","PKM"]
push!(ax,[Axis(f[1,i],title=tits[i]) for i in 1:3])
[push!(ax,[Axis(f[j,i]) for i in 1:3]) for j in 2:3]
display(GLMakie.Screen(),f)

model=creb
mod_name="CREB"
u0 = [:C1 => 10, :C2 => 0]
ps = [:V_x => 0.1, :V_y => 0.01, :K_x => 5, :K_y => 10, :k_dx => 0.04, :k_dy => 0.01, :r_bas_x => 0.003, :r_bas_y => 0.002, :Ω => 10]
ode,sde,jprob,eprob=make_models(model,u0,tspan,ps)
solve_plot(model,mod_name,ode,tspan,tstops,ax[1][1];plot_symbs=[:C1,:C2])

u0 = [:C1 => 10, :C2 => 0]
ps = [:V_x => 0.2, :V_y => 0.01, :K_x => 5, :K_y => 10, :k_dx => 0.04, :k_dy => 0.01, :r_bas_x => 0.003, :r_bas_y => 0.002, :Ω => 10]
ode,sde,jprob,eprob=make_models(model,u0,tspan,ps)
solve_plot(model,mod_name,ode,tspan,tstops,ax[2][1];plot_symbs=[:C1,:C2])

u0 = [:C1 => 10, :C2 => 0]
ps = [:V_x => 0.4, :V_y => 0.01, :K_x => 5, :K_y => 10, :k_dx => 0.04, :k_dy => 0.01, :r_bas_x => 0.003, :r_bas_y => 0.002, :Ω => 10]
ode,sde,jprob,eprob=make_models(model,u0,tspan,ps)
solve_plot(model,mod_name,ode,tspan,tstops,ax[3][1];plot_symbs=[:C1,:C2])

model=erk
u0 = [:E => 0, :M => 0, :P=>0, :X=>0]
ps = [:K => 0.05, :δ=>50, :γ=>1, :U=>0.0]
ode,sde,jprob,eprob=make_models(model,u0,tspan,ps)
v,s=solve_plot(model,"ERK",ode,tspan,tstops,ax[1][2];plot_symbs=[:E,:M])

u0 = [:E => 0, :M => 0, :P=>0, :X=>0]
ps = [:K => 2, :δ=>50, :γ=>1, :U=>0.0]
ode,sde,jprob,eprob=make_models(model,u0,tspan,ps)
v,s=solve_plot(model,"ERK",ode,tspan,tstops,ax[2][2];plot_symbs=[:E,:M])

u0 = [:E => 0, :M => 0, :P=>0, :X=>0]
ps = [:K => 10, :δ=>50, :γ=>1, :U=>0.0]
ode,sde,jprob,eprob=make_models(model,u0,tspan,ps)
v,s=solve_plot(model,"ERK",ode,tspan,tstops,ax[3][2];plot_symbs=[:E,:M])

model=pkm
u0 = [:PKM => 0.997, :PKMa => 0.003, :FActin=>1, :FActina=>0, :RNA=>1, :RNAa=>0]
ps=[:j_1 => 10, :j_2 => 0.05, :j_3 => 0.5, :j_4 => 0.16,:τ_1=>1500,:τ_2 =>0.5,:τ_3 => 50,:U=>0]
ode,sde,jprob,eprob=make_models(model,u0,tspan,ps;jump_p=false)
solve_plot(model,"PKMζ",ode,tspan,tstops,ax[1][3];plot_symbs=[:PKMa,:FActina,:RNAa])


u0 = [:PKM => 0.997, :PKMa => 0.003, :FActin=>0, :FActina=>0, :RNA=>1, :RNAa=>0]
ps=[:j_1 => 80, :j_2 => 0.05, :j_3 => 0.5, :j_4 => 0.16,:τ_1=>1500,:τ_2 =>0.5,:τ_3 => 50,:U=>0]
ode,sde,jprob,eprob=make_models(model,u0,tspan,ps;jump_p=false)
solve_plot(model,"PKMζ",ode,tspan,tstops,ax[2][3];plot_symbs=[:PKMa,:FActina,:RNAa])


u0 = [:PKM => 0.997, :PKMa => 0.003, :FActin=>1, :FActina=>0, :RNA=>1, :RNAa=>0]
ps=[:j_1 => 80, :j_2 => 0.05, :j_3 => 0.5, :j_4 => 0.16,:τ_1=>1500,:τ_2 =>0.5,:τ_3 => 50,:U=>0]
ode,sde,jprob,eprob=make_models(model,u0,tspan,ps;jump_p=false)
solve_plot(model,"PKMζ",ode,tspan,tstops,ax[3][3];plot_symbs=[:PKMa,:FActina,:RNAa])
