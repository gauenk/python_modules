import numpy as np
import numpy.random as npr

def const_rate(tau,const=1):
    return const*tau

def const_hazard(tau):
    return 1./tau

def weibull_rate(tau,shape=1,scale=1):
    mean = (tau/scale)**shape
    return const*tau

def weibull_hazard(tau,shape=1,scale=1):
    rv = Weibull({'shape':shape,'scale':scale},
                 is_hazard_rate=True)
    return rv

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
        self.sim_via_thinning = False
        self.sim_via_rejection_sampling = False

    # nonstationary simulation check
    def nonstation_sim_check(self):
        """
        ! verify we have parmaters to simulate
        """
        # simulation using thinning
        bool_simA = self.rate_fxn is not None \
            and self.rate_ub is not None
        self.sim_via_thinning = bool_simA
        # simulation using rejection sampling
        """
        Why does sampling rate funxtion require a time window?
        - not true for non-stationary i don't think
        - this is true for stationary.
        Why?
        no.... for stationary the rate function is exponential?
        but this is not the same ~rate function~ as the constant \lambda...
        So then what is the A_{v,v'} term?

        We don't actually sample the "rate_fxn". rather we sample the 
        associated distribution function of the rate_fxn.
        e.g. rate_fxn(t) = f(t) / S(t); instead we simulate the 
        rate function for the associated distribution f(t). Thus for 
        "rate_fxn_simulation" this is actually simulating the associated 
        distribution function.
        """

        bool_simB = self.rate_fxn_sampler is not None \
            and self.parameter_measure is not None
        self.sim_via_rejection_sampling = bool_simA

    # stationary parameters
    def constant_parameter_measure(self,ts,te):
        tau = te - ts
        return self.lambda0 * tau

    def constant_rate_fxn_sampler(self,ts,te):
        # this should be an exponential then...
        tau = te - ts
        return npr.uniform(0,tau)

    def constant_rate_fxn(self,tau):
        return self.lambda0

    def sample(self,*args,**kwargs):
        print("REFACTORED CODE. THIS MAY NOT BE THE SAMPLING YOU WANT!")
        if 'sample_type' not in kwargs:
            raise ValueError("No 'sample_type' provided. We need to know what we are sampling.")
        sample_type = kwargs['sample_type']
        if sample_type == 'hold_time':
            return sample_hold_time(self,*args,**kwargs)
        elif sample_type == 'conditional_N':
            return sample_conditional_N(self,*args,**kwargs)
        elif sample_type == 'interval':
            return sample_interval(self,*args,**kwargs)
        else:
            raise ValueError("Uknown 'sample_type' = [{}]".format(sample_type))

    def sample_hold_time(self,n=1):
        """
        Samples the time to the n^th event

        Homogenous Poisson Process
        -> time to kth event is Erlang distribution 
        -> time to first event is exponential

        Nonhomogenous Poisson Process
        -> ???
        """
        # TODO: n > 1
        if n > 1:
            raise ValueError("I've not done this yet...")
        if self.stationary:
            scale = 1/self.lambda0
            return npr.exponential(scale=scale,size=1)
        else:
            return self.sample_hold_time_ns(n)

    def sample_hold_time_ns(self,n=1):
        """
        sample n^th event times of a nonstationary (ns) poisson process
        """
        if self.sim_via_thinning:
            while(True):
                u = npr.uniform(0,1)
                w = -np.log(u/self.rate_ub)
                D = npr.uniform(0,1)
                if D <= self.rate_fxn(w)/self.rate_ub:
                    return t
        elif self.sim_via_rejection_sampling:
            tau = None
            return self.rate_fxn_sampler(0,tau)
        raise ValueError("No sampling method specified")
        return None

    def sample_interval(self,tau,offset=0):
        if tau <= 0:
            raise ValueError("Hold time must be positive. [{}]".format(tau))
        # [nonstationary] int_0^\tau \lambda(t) dt __OR__ \lambda * t [stationary]
        mean = self.parameter_measure(0,tau)
        N = npr.poisson( lam = mean )
        # --> rejection sampling <--
        samples = []
        i = 0
        while (i < N):
            sample = self.rate_fxn_sampler(0,tau)
            if sample <= tau:
                samples.append(sample)
                i += 1
        samples = np.array(sorted(samples))
        # handles the offset by interval
        if len(samples) > 0: 
            samples = samples + offset 
        return samples

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

    def weibull_mean(self,t_start,t_end,current_state,state):
        # compute the Poisson process mean for weibull hazard function
        curr_state_index = self.state_space.index(current_state)
        next_state_index = self.state_space.index(state)
        shape = self.mean_function_params['shape'][curr_state_index][next_state_index]
        scale = self.mean_function_params['scale'][curr_state_index][next_state_index]
        # print(scale,t_end,t_start,shape)
        # print(scale**shape)
        mean = (t_end/scale)**shape - (t_start/scale)**shape
        # print(mean)
        return mean

    def update_parameters(self,parameters):
        if 'shape' in parameters.keys():
            self.mean_function_params['shape'] = parameters['shape']
        if 'scale' in parameters.keys():
            self.mean_function_params['scale'] = parameters['scale']

    def poisson_likelihood(self,mean,k):
        return mean**k * np.exp(-mean) / np.math.factorial(k)



class CRP():
    
    def __init__(self,alpha=30,n=None,c=None):
        # init
        self.alpha = alpha
        self.n = 0
        self.c = []
        self.sizes = []

        # error checking
        if n is not None:
            if len(c) != n:
                raise InputError("Number of clusters must be equal to cluster assigment length")
            self.n = n
            self.c = c
            self.sizes = self.bincount(c)

    def sample(self,n=1):
        clusters = []
        for i in range(n):
            cluster = self.single_sample()
            clusters.append(cluster)
        return clusters

    def single_sample(self):
        denom = self.n - 1 + self.alpha
        if denom > 0:
            prob = self.alpha / denom
        else:
            prob = 1.0
        # coin flip
        acc = npr.uniform(0,1) < prob
        if acc or self.n == 0: # start a new table
            self.sizes += [0]
            cluster = len(self.sizes) - 1 # yes "n + 1" but "-1" since 0 indexed
        else: # join an old one.
            probs = self.sizes / np.sum(self.sizes)
            cluster = np.where(npr.multinomial(1,probs))[0][0]
        self.sizes[cluster] += 1
        self.n += 1
        self.c += [cluster]
        return cluster


