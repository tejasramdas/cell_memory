using Markdown
md"""
## All models
"""



begin
	simple_switch_crn = @reaction_network begin
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

begin 
	crick_crn = @reaction_network begin
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

begin
	mts_crn = @reaction_network begin
		@species M1(t) M2(t) M3(t)
		@parameters t_on t_off l_on k_l1=1 k_l2=0 k_l3=0
		@default_noise_scaling 0.2
		@discrete_events begin
			((t == t_on)) => [l ~ l_on]
			((t == t_off)) => [l ~ 0.0]
		end
		((k_f1,k_f2,k_f3),(k_b1,k_b2,k_b3)), ∅ <--> (M1, M2, M3)
    (k_f12*k_f2,k_f13*k_f3), M1 --> (M1+M2,M1+M3)
		k_f23*k_f3, M2 --> M2+M3
    (l*k_l1*k_f1,l*k_l2*k_f2,l*k_l3*k_f3),  ∅ --> (M1,M2,M3)
	end
end

begin
	mts_simp = @reaction_network begin
		@species M1(t) M2(t)
		@parameters t_on t_off l_on k_l1=1 k_l2=0 k_l3=0 l
		@default_noise_scaling 0.2
		@discrete_events begin
			((t == t_on)) => [l ~ l_on]
			((t == t_off)) => [l ~ 0.0]
		end
		((k_f1,k_f2),(k_b1,k_b2)), ∅ <--> (M1, M2)
    k_f12*k_f2, M1 --> M1+M2
	end
end

begin
	hillcreb(C1,C2,K_x,K_y,V,Ω) = hillar(C1/(√K_x *Ω),C2/(√K_y *Ω), V*Ω, 1, 2)
	creb_crn = @reaction_network begin
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

begin
	erk_crn = @reaction_network begin
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

begin
	pkm_crn = @reaction_network begin
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

epi_simple = @reaction_network begin
	(kkw1+kkm1,kke1+delta1+kkke1), D <--> Dm1
	(kkw1+kkm1,kke1+delta1+kkke1), D <--> Dm2
end

epi = @reaction_network begin
	(kw1010+kw1+kmprime+kmprime,deltaprime+ktprime+ktprimeact), D <--> D1
	(kw20+km+km+kmbar+kmbar,delta+ke+keact), D1 <--> D12
	(kw20+kw2+km+km+kmbar+kmbar,delta+ke+keact), D <--> D2
	(kw10+kmprime+kmprime,deltaprime+ktprime+ktprimeact), D2 <--> D12
	(kwa0+kwa+kma, delta+kea+keacta+keacta+keacta+keacta), D <--> Da
end
