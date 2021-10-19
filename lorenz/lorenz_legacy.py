import numpy as np
from scipy.integrate import solve_ivp

def get_lorenz96_sim():
    def AnDA_Lorenz_96(S,t,F,J):
        """ Lorenz-96 dynamical model. """
        x = np.zeros(J);
        x[0] = (S[1]-S[J-2])*S[J-1]-S[0];
        x[1] = (S[2]-S[J-1])*S[0]-S[1];
        x[J-1] = (S[0]-S[J-3])*S[J-2]-S[J-1];
        for j in range(2,J-1):
            x[j] = (S[j+1]-S[j-2])*S[j-1]-S[j];
        dS = x.T + F;
        return dS


    class GdCls:
        model = 'Lorenz_96'

        class parameters:
            F = 8
            J = 40
        dt_integration = 0.05 # integration time
        dt_states = 1 # number of integration times between consecutive states (for xt and catalog)
        dt_obs = 4 # number of integration times between consecutive observations (for yo)
        var_obs = np.random.permutation(parameters.J)[0:20] # indices of the observed variables
        nb_loop_train = 10**4 # size of the catalog
        nb_loop_test = 10 # size of the true state and noisy observations
        sigma2_catalog = 0   # variance of the model error to generate the catalog   
        sigma2_obs = 2 # variance of the observation error to generate observations

    class time_series:
      values = 0.
      time   = 0.

    # 5 time steps (to be in the attractor space)
    GD = GdCls()    
    x0 = GD.parameters.F*np.ones(GD.parameters.J);
    x0[int(np.around(GD.parameters.J/2))] = x0[int(np.around(GD.parameters.J/2))] + 0.01;

    #S = odeint(AnDA_Lorenz_96,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));
    S = solve_ivp(
        fun=lambda t,y: AnDA_Lorenz_96(y,t,GD.parameters.F,GD.parameters.J),
        t_span=[0.,5+1e-6],
        y0=x0,
        first_step=GD.dt_integration,
        t_eval=np.arange(0,5+1e-6,GD.dt_integration),
        method='RK45'
    )
    x0 = S.y[:,-1]

    # generate true state (xt)
    #S = odeint(AnDA_Lorenz_96,x0,np.arange(0.01,GD.nb_loop_test+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));       
    tt = np.arange(GD.dt_integration, GD.nb_loop_train*GD.dt_integration+1e-6, GD.dt_integration)
    S = solve_ivp(
        fun=lambda t,y: AnDA_Lorenz_96(y, t, GD.parameters.F, GD.parameters.J),
        t_span=[GD.dt_integration, GD.nb_loop_train*GD.dt_integration+1e-6],
        y0=x0,
        first_step=GD.dt_integration,
        t_eval=tt,
        method='RK45'
    )
    S = S.y.transpose()

    return S, tt

