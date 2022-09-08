import numpy as np
import pandas as pd

def kl_mvn(m0, S0, m1, S1):
    """
    https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
    The following function computes the KL-Divergence between any two 
    multivariate normal distributions 
    (no need for the covariance matrices to be diagonal)
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    - accepts stacks of means, but only one S0 and S1
    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    # 'diagonal' is [1, 2, 3, 4]
    tf.diag(diagonal) ==> [[1, 0, 0, 0]
                          [0, 2, 0, 0]
                          [0, 0, 3, 0]
                          [0, 0, 0, 4]]
    # See wikipedia on KL divergence special case.              
    #KL = 0.5 * tf.reduce_sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=1)   
                if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))                               
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 

def fit_mv_normal(n_array):
    """
    Calculate multivariate normal distribution
    """
    df = pd.DataFrame(n_array)
    mean = np.mean(df, axis=0)
    cov = np.cov(df, rowvar=0)
    return (mean,cov)

def distance_to_aggregator(parties):    
    """
    Calculate distance to average fused aggregator
    - input: list of parties in nummpy array
    - ouput: list of distance in scalar
    """
    means = []
    covs = []
    party_num = len(parties)
    
    assert(party_num > 0)
    i = 0
    for party in parties :
        mean, cov = fit_mv_normal(party)
        means.append(mean)
        covs.append(cov)
        i = i+1
    
    avg_mean = sum(means)/party_num
    avg_cov = sum(covs)/party_num
    
    kl_distance = [kl_mvn(means[i],covs[i],avg_mean,avg_cov) for i in range(party_num) ]
    
    return kl_distance

def distance_to_global(prob_list):
    """
    Calculate distance to average fused aggregator
    - input: list of probability distribution of parties [mean,cov]
    - ouput: list of distance in scalar
    """
    party_num = len(prob_list)
    assert(party_num > 0)
    
    means = []
    covs = []
    for prob in prob_list :
    
        means.append(prob[0])
        covs.append(prob[1])
    
    avg_mean = sum(means)/party_num
    avg_cov = sum(covs)/party_num
    
    kl_distance = [kl_mvn(means[i],covs[i],avg_mean,avg_cov) for i in range(party_num) ]
    
    return kl_distance