import numpy as np
import scipy.stats as stats
from scipy.stats import mvn

class truncMVN(object):
    """
    Class to calculate PDF of a 2D truncated multivariate normal distribution.
    
    Uses method of Alan Genz to calculate MVN CDF, implemented in scipy.
    
    low_int_limit: lower limit for *integration* of the CDFs. This is different than the lower
    limit of the bound we want to normalize! Setting to a value several SDs lower than mean should be fine.
    
    low: lower limit for the bound that we want to normalize over. Again, different from the
    lower integration limit!
    
    high: upper limit for the bound that we want to normalize over
    
    calc_norm: If true, calculate normalization at initialization. 
    """
    def __init__(self, mean=None,cov=None,low=None,high=None,low_int_limit=np.array([-10.,-10.]),calc_norm=False):                    
        self.low_int_limit = low_int_limit
        self.low = low
        self.high = high
        
        if calc_norm:
            self.norm = self.normalize(mean,cov,high,low)
            
    
    def normalize(self,mean=None,cov=None,low=None,high=None):
        """
        Calculate normalization term for the truncated MVN.
        
        Involves calculation of four CDF terms:
        P(4) - P(3) - P(2) + P(1)
        """
        #calculate CDF of full region
        cdf_4,i = mvn.mvnun(self.low_int_limit,high,mean,cov)
        
        #calculate CDFs of outside regions
        cdf_3,i = mvn.mvnun(self.low_int_limit,np.array([high[0],low[1]]),mean,cov)
        cdf_2,i = mvn.mvnun(self.low_int_limit,np.array([low[0],high[1]]),mean,cov)
        
        #calculate CDF of lower-left corner region
        cdf_1,i = mvn.mvnun(self.low_int_limit,low,mean,cov)
        
        reg_prob = (cdf_4 - cdf_3 - cdf_2 + cdf_1)
                
        return 1./reg_prob
            
        
    def logpdf(self,data,mean=np.array([]),cov=np.array([]),re_norm=True,approx=True):
        """
        Calculate the logpdf for a truncated MVN
        
        data = 2 x n array containing data to calc PDFs for.
        mean: 2 x 1 array value for MVN mean
        cov:  2x2 covariance matrix
        
        re_norm: if true, re-calculate normalization of the pdf
        
        approx: if sigmas are small, normalization is very
        close to 1 and might throw errors, so just skip. 
        """
        lpdf = stats.multivariate_normal.logpdf(data,mean=mean,cov=cov)
        
        
        if re_norm:
            #if the covaraince terms are really small, re-normalization
            #seems to throw NaNs and we probably don't need to anyway
            #could be dangerous.
            if (cov[0,0] < 1e-2) or (cov[1,1] < 1e-2) and approx:
                norm = 1.
                
            else:
                norm = self.normalize(mean,cov,self.low,self.high)
        else:
            norm = self.norm
            
        return np.log(norm) + lpdf
        
        
    def pdf(self,data,mean=np.array([]),cov=np.array([]),re_norm=True):
        """
        Calculate the pdf for a truncated MVN
        
        data = 2 x n array containing data to calc PDFs for.
        mean: 2 x 1 array value for MVN mean
        cov:  2x2 covariance matrix
        """
        pdf = stats.multivariate_normal.logpdf(data,mean=mean,cov=cov)
        
        if re_norm:
            norm = self.normalize(mean,cov,self.low,self.high)
        else:
            norm = self.norm
            
        return norm * pdf