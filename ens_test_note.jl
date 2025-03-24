### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ af7bcb2c-0757-11f0-10ea-af37b4a77c5e
using Pkg

# ╔═╡ 36875680-4d09-44fd-9c55-e6f3b47dc2df
Pkg.activate(".")

# ╔═╡ be3e028a-bb08-4aaf-974b-8b22fd33319e
using Catalyst,DifferentialEquations,WGLMakie

# ╔═╡ 1a5a9378-3393-4f3c-98a1-6464050dcc23
html"""
<style>main {max-width: 100%; overflow:hidden; padding-right: 5%;}</style>
<style>pluto-editor main {margin-right: 0;}</style>
<style>pluto-output.scroll_y {max-height: 2000px;}</style>
"""

# ╔═╡ d7155949-de4b-4d11-b6b7-21fc6d377e31
begin
	simple_switch = @reaction_network begin
		@species M0(t) M1(t)
		@parameters t_on t_off l_on
		@default_noise_scaling 0.1
		@discrete_events begin
			(t==t_on) => [l ~ l_on]
			(t==t_off) => [l ~ 0.0]
		end
		(k_f1,k_b1), M0 <--> M1
		l, M0 --> M1
	end
end

# ╔═╡ 3d7e5c92-497d-4587-8ccc-57dd8f335160
begin
	t_on=25.0
	t_off=150.0
end

# ╔═╡ a93dd3b1-445d-4412-8a37-5ba982082d33
begin
	u0=[:M0=>100.0,:M1=>0.0]
	ps=[:k_f1=>0.01,:k_b1=>0.5,:l=>0.0,:t_on=>t_on,:t_off=>t_off,:l_on=>0.1]
	tspan=(0.0,200.0)
	u0_int=[:M0=>100,:M1=>0]
end

# ╔═╡ 65bca955-6949-4448-9e83-cf761c7de6df
begin
	ode=ODEProblem(simple_switch,u0,tspan,ps)
	sde=SDEProblem(simple_switch,u0,tspan,ps)
	jinput = JumpInputs(simple_switch,u0_int,tspan,ps)
	jprob=JumpProblem(jinput)
end

# ╔═╡ 149c9bcb-4277-46de-94c4-3d2fd96b32e8
begin
	probs=[ode,sde,jprob]
	eprobs=[EnsembleProblem(probs[i]) for i in 1:3]
end

# ╔═╡ baec605a-2cdd-4582-ad2a-955aa8d870fc
begin
	f=Figure(size=(1500,500))
	ax=[Axis(f[1,i]) for i in 1:3]
	[ax[i].title=["ODE","SDE","Jump"][i] for i in 1:3]
end

# ╔═╡ 2f936541-fa02-4603-807a-e021140d9815
f

# ╔═╡ bc799ae0-e094-42d2-bc9c-457efd14cf27
md"""
### Single simulation works fine 
"""

# ╔═╡ 29a5ab4b-cb03-4e3d-9709-b5a8e9ffdffa
for i in 1:50
    for j in 1:3
        empty!(ax[j]) 
        sol=solve(probs[j];tstops=[t_on,t_off])
        lines!(ax[j],0:0.1:200,sol.(0:0.1:200,idxs=2),color=:blue)
    end
    sleep(0.01)
end

# ╔═╡ 711948cb-473e-41b3-89b4-314187113614
md"""
### Ensemble simulation introduces random perturbations
For example, the ODE simlations should be identical across the ensemble
"""

# ╔═╡ 54d34ca1-7eaf-47a6-99c8-220b37bf985b
for i in 1:5
    for j in 1:3
        empty!(ax[j]) 
        if j==3
            sol=solve(eprobs[j],SSAStepper();trajectories=100,tstops=[t_on,t_off])
        else
            sol=solve(eprobs[j];trajectories=100,tstops=[t_on,t_off])
        end
        for k in 1:100
            lines!(ax[j],0:0.1:200,sol[k].(0:0.1:200,idxs=2),color=:blue,alpha=0.1)
        end
    end
    sleep(0.05)
end

# ╔═╡ Cell order:
# ╠═af7bcb2c-0757-11f0-10ea-af37b4a77c5e
# ╠═1a5a9378-3393-4f3c-98a1-6464050dcc23
# ╠═36875680-4d09-44fd-9c55-e6f3b47dc2df
# ╠═be3e028a-bb08-4aaf-974b-8b22fd33319e
# ╠═d7155949-de4b-4d11-b6b7-21fc6d377e31
# ╠═3d7e5c92-497d-4587-8ccc-57dd8f335160
# ╠═a93dd3b1-445d-4412-8a37-5ba982082d33
# ╠═65bca955-6949-4448-9e83-cf761c7de6df
# ╠═149c9bcb-4277-46de-94c4-3d2fd96b32e8
# ╠═baec605a-2cdd-4582-ad2a-955aa8d870fc
# ╠═2f936541-fa02-4603-807a-e021140d9815
# ╠═bc799ae0-e094-42d2-bc9c-457efd14cf27
# ╠═29a5ab4b-cb03-4e3d-9709-b5a8e9ffdffa
# ╠═711948cb-473e-41b3-89b4-314187113614
# ╠═54d34ca1-7eaf-47a6-99c8-220b37bf985b
