def grubbs_train(timeseries):
    
    series = scipy.array([x for x in timeseries])    
    len_series = len(series)+1 + 40
    threshold = scipy.stats.t.isf(.01 / (2 * len_series), len_series - 2)
    threshold_squared = threshold * threshold
    grubbs_score = ((len_series - 1) / np.sqrt(len_series)) * np.sqrt(threshold_squared / (len_series - 2 + threshold_squared))
    return grubbs_score

def grubbs_test(timeseries):
    """
    A timeseries is anomalous if the Z score is greater than the Grubb's score.
    """
    series = scipy.array([x for x in timeseries])
    stdDev = np.std(series)  
    mean = np.mean(series)
    tail_average = tail_avg(timeseries)
    z_score = (tail_average - mean) / stdDev
    return z_score    
