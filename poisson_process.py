import numpy as np
import numpy.random as npr

class PoissonProcess(object):
    """
    Example:
    
    Homogenous (Stationary) Poisson Process
    p = PoissonProcess('s',3)
    

    """


    def __init__(self,rate_fxn_type,rate_fxn,
                 rate_fxn_sampler = None,
                 parameter_measure = None,
                 rate_ub = None):
        """
        rate_fxn:
        the "\lambda(t)" term; this is \lambda for Stationary Poisson Process
        
        rate_fxn_type:
        this dictates what type of Poisson Process we have

        parameter_measure:
        for non-stationary Poisson Processes,
        we have \Lambda(t) = \int_0^t \lambda(s) ds
        """

        # are we stationary?
        bool_stationary = rate_fxn_type in ['c','s','h']
        bool_stationary = bool_stationary or rate_fxn_type == 'constant'
        bool_stationary = bool_stationary or rate_fxn_type == 'stationary'
        self.stationary = bool_stationary

        if self.stationary:
            self.lambda0 = rate_fxn
            self.parameter_measure = self.constant_parameter_measure
            self.rate_fxn_sampler = self.constant_rate_fxn_sampler
            self.rate_fxn = self.constant_rate_fxn
        else:
            self.parameter_measure = parameter_measure
            self.rate_fxn_sampler = rate_fxn_sampler 
            self.rate_fxn = rate_fxn
        self.rate_ub = rate_ub
        self.sim_interval_thinning = True
        # self.sim_times_via_rate = False
        # self.sim_N_via_param_msr = False
        self.sim_traj_weird = False
        self.nonstation_sim_check()

    # nonstationary simulation check
    def nonstation_sim_check(self):
        """
        ! verify we have parmaters to simulate
        """
        # simulation using thinning
        self.sim_interval_thinning = self.rate_fxn is not None \
            and self.rate_ub is not None
        # simulation using rejection sampling
        # self.sim_times_via_rate = self.rate_fxn_sampler is not None
        # self.sim_N_via_param_msr = self.parameter_measure is not None
        # self.sim_traj_weird = self.rate_fxn_sampler is not None\
        #     and self.parameter_measure is not None

    # stationary parameters
    def constant_parameter_measure(self,ts,te):
        tau = te - ts
        return self.lambda0 * tau

    def constant_rate_fxn_sampler(self,tau):
        # this should be an exponential then... 
        # AND it is since the order stats of Exp
        # is the same as a uniform. this is more efficient
        return npr.uniform(0,tau)

    def constant_rate_fxn(self,tau):
        return self.lambda0

    def sample(self,*args,**kwargs):
        if 'sample_type' not in kwargs:
            raise ValueError("No 'sample_type' provided. We need to know what we are sampling.")
        sample_type = kwargs['sample_type']
        del kwargs['sample_type']
        if sample_type == 'hold_time':
            return self.sample_hold_time(*args,**kwargs)
        elif sample_type == 'trajectory':
            return self.sample_trajectory(*args,**kwargs)
        else:
            raise ValueError("Uknown 'sample_type' = [{}]".format(sample_type))

    def sample_hold_time(self,n=1):
        """
        Samples the time to the n^th event

        Homogenous Poisson Process
        -> time to first event is exponential
        -> time to kth event is Erlang (Gamma) distribution 
           -> sum of k exponentials
        """
        if n > 1:
            raise ValueError("I've not done this yet...")
        if self.stationary:
            scale = 1/self.lambda0
            return npr.exponential(scale=scale,size=1)
        else:
            return self.sample_hold_time_ns(n=1)

    def sample_hold_time_ns(self,n=1):
        """
        returns the first event time from a non-stationary poisson process
        """
        if self.sim_interval_thinning:
            while(True):
                u = npr.uniform(0,1)
                w = -np.log(u)/self.rate_ub # v.s. np.log(u/self.rate_ub)
                D = npr.uniform(0,1)
                if D <= self.rate_fxn(w)/self.rate_ub:
                    return w
        elif self.sim_times_via_rate:
            raise ValueError("No sampling method specified")
        return None

    def sample_trajectory(self,T):
        if self.stationary:
            return self.sample_trajectory_stationary(T)
        elif self.sim_traj_weird:
            return self.sample_trajectory_weird(T)
        elif self.sim_interval_thinning:
            return self.sample_trajectory_thinning(T)
        else:
            raise ValueError("No simulation method available")

    def sample_trajectory_stationary(self,T):
        N = npr.poisson(self.lambda0*T)
        samples = npr.uniform(0,T,size=N)
        samples = np.sort(samples)
        return samples

    def sample_trajectory_weird(self,T):
        raise NotImplemented("We compare this to Kent's smjp code.")

    def sample_trajectory_thinning(self,T):
        # requires bounded rate (intensity or hazard) function
        samples = []
        t = 0
        nr = 0
        ubpp = PoissonProcess('c',self.rate_ub)
        while(t < T):
            s_h = ubpp.sample( sample_type = 'hold_time')
            s = t + s_h
            log_u = np.log(npr.uniform(0,1))
            log_rate_fxn = np.log(self.rate_fxn(s))
            log_rate_ub = np.log(self.rate_ub)
            if log_u <= log_rate_fxn - log_rate_ub:
                samples.append(s)
            else:
                nr += 1
            t = s
        print("Number of Rejections: {}".format(nr))
        print("Number of Acceptances: {}".format(len(samples)))
        if samples[-1] > T: samples.pop()
        return np.squeeze(np.array(samples))

    def l(self,*args,**kwargs):
        return self.likelihood(*args,**kwargs)

    def likelihood(self,interval,current_state,is_poisson_event=True):
        tau = interval[1]
        """
        interval = ["current_hold_time", "next_hold_time"]
        tau \neq delta w.... we

        delta w_i = interval[1] - interval[0]
        we want the total hold time though...
        """
        # A_{s,s'}(\tau) is the hazard function
        # = [\sum_{s'\in S} A_{s,s'}(\tau)] * exp\{ \int_{l_i}^{l_i + \delta w_i} A_s(\tau) \}
        # -----------------------------
        # -=-=-=-> version 1 <-=-=-=-=-
        # -----------------------------
        if not is_poisson_event:
            log_exp_term = 0
            for next_state in self.state_space:
                log_exp_term += self.mean_function(interval[0], interval[1], current_state,next_state)
            return np.exp(-log_exp_term)

        front_term = 0
        log_exp_term = 0
        for next_state in self.state_space:
            front_term += self.hazard_function(tau,current_state,next_state)
            log_exp_term += self.mean_function( interval[0], interval[1],
                                                current_state,
                                                next_state)
        # print("front_term",front_term)
        # print("exp_term",np.exp(-log_exp_term))
        likelihood = front_term * np.exp(-log_exp_term)
        return likelihood


    def update_parameters(self,parameters):
        if 'shape' in parameters.keys():
            self.mean_function_params['shape'] = parameters['shape']
        if 'scale' in parameters.keys():
            self.mean_function_params['scale'] = parameters['scale']

    def poisson_likelihood(self,mean,k):
        return mean**k * np.exp(-mean) / np.math.factorial(k)

    def plot_intensity(self,ax,T,lcm='kx-',G = 100):
        xgrid = np.linspace(0,T,G)
        intensity = [self.rate_fxn(x) for x in xgrid]
        handle, = ax.plot(xgrid,intensity,lcm)
        return handle

    def __str__(self):
        info = "Poisson Process Information\n"
        if self.stationary:
            info += self._str_stationary()
        else:
            info += self._str_nonstationary()
        return info

    def _str_stationary(self):
        info = "Stationary\n"
        info += "lambda0: {}\n".format(self.lambda0)
        return info

    def _str_nonstationary(self):
        info = "Non Stationary\n"
        return info

