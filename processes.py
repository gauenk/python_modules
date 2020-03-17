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
        print(rate_fxn_type,self.stationary)

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
        self.sim_interval_thinning = False
        # self.sim_times_via_rate = False
        # self.sim_N_via_param_msr = False
        self.sim_interval_rs = False
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
        self.sim_interval_rs = self.rate_fxn_sampler is not None\
            and self.parameter_measure is not None

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
        print("REFACTORED CODE. THIS MAY NOT BE THE SAMPLING YOU WANT!")
        if 'sample_type' not in kwargs:
            raise ValueError("No 'sample_type' provided. We need to know what we are sampling.")
        sample_type = kwargs['sample_type']
        del kwargs['sample_type']
        if sample_type == 'hold_time':
            return self.sample_hold_time(*args,**kwargs)
        elif sample_type == 'conditional_N':
            return self.sample_conditional_N(*args,**kwargs)
        elif sample_type == 'interval':
            return self.sample_interval(*args,**kwargs)
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
            return self.rate_fxn_sampler()

    def sample_hold_time_ns(self,n=1):
        """
        sample n^th event times of a nonstationary (ns) poisson process
        """
        if self.sim_interval_thinning:
            while(True):
                u = npr.uniform(0,1)
                w = -np.log(u/self.rate_ub)
                D = npr.uniform(0,1)
                if D <= self.rate_fxn(w)/self.rate_ub:
                    return w
        elif self.sim_times_via_rate:
            raise ValueError("No sampling method specified")
        return None

    def sample_interval(self,tau,offset=0):
        if self.stationary or self.sim_interval_rs:
            return self.sample_interval_rs(tau,offset)
        elif self.sim_interval_thinning:
            return self.sample_interval_thinning(tau,offset)            
        else:
            raise ValueError("No simulation method available")

    def sample_interval_thinning(self,tau,offset=0):
        # requires bounded rate (intensity or hazard) function
        print("sit")
        samples = []
        t = 0
        s = 0
        while(s < tau):
            # sample exp
            u = npr.uniform(0,1)
            w = -np.log(u)/self.rate_ub
            s = s + w
            D = npr.uniform(0,1)
            if D <= self.rate_fxn(s)/self.rate_ub:
                samples.append(s)
        if samples[-1] > tau: # remove "too long" endpoint
            samples.pop()
        return np.array(samples)

    def sample_interval_rs(self,tau,offset=0):
        if tau <= 0:
            raise ValueError("Hold time must be positive. [{}]".format(tau))
        # [nonstationary] int_0^\tau \lambda(t) dt __OR__ \lambda * t [stationary]
        mean = self.parameter_measure(0,tau)
        print("mean: {}".format(mean))
        N = npr.poisson( lam = mean )
        # --> rejection sampling <--
        samples = []
        i = 0
        while (i < N):
            if self.stationary:
                sample = self.rate_fxn_sampler(tau) # stationary
            else:
                sample = self.rate_fxn_sampler()
            shape = 2
            sample = npr.weibull(shape,size=1)[0]
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


