using Catalyst, DifferentialEquations, WGLMakie


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

t_on=25.0
t_off=150.0
u0=[:M0=>100.0,:M1=>0.0]
ps=[:k_f1=>0.01,:k_b1=>0.5,:l=>0.0,:t_on=>t_on,:t_off=>t_off,:l_on=>0.1]
tspan=(0.0,200.0)
u0_int=[:M0=>100,:M1=>0]

ode=ODEProblem(simple_switch,u0,tspan,ps)
sde=SDEProblem(simple_switch,u0,tspan,ps)
jinput = JumpInputs(simple_switch,u0_int,tspan,ps)
jprob=JumpProblem(jinput)

f=Figure(size=(1500,500))
ax=[Axis(f[1,i]) for i in 1:3]

probs=[ode,sde,jprob]
eprobs=[EnsembleProblem(probs[i]) for i in 1:3]


for i in 1:50
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
        ax[j].title="Time: $i"
    end
    sleep(0.05)
end

