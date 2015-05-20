# Submission notes

Primarily keep track of cv results, model parameters, and features
used for each submission.

## CV results

13.
n_st=500
0.91540721992 0.0203362966484
0.954939205096 0.00133639367936

n_est=4000
0.915271602114 0.0198226221489
0.953627599105 0.00242965319325


14.
n_est=500
outlier removed:
n_features=10
0.929231850808 0.0156712434759
0.958162777538 0.001760586993


n_est=1000
outlier removed:
n_features=22
0.940344774491 0.0194237716926
0.959678707813 0.000498810944577

n_est = 1000
['bba_3', 'bba_2', 'num_bids', 'bba_1', 'num_ips', 'phone46', 'au',
'phone143', 'phone28', 'th', 'phone13', 'phone17', 'phone62', 'phone290',
'phone157', 'phone479', 'phone237', 'phone346', 'phone248', 'phone119', 
'phone56', 'phone122']

0.927225232133 0.0208572435834
0.959880471005 0.00168220270904

15. 
n_est = 1000
chose 10*4 most important features from each category by ET
['num_bids', 'au', 'num_aucs', 'bba_15', 'num_ips', 'bba_14', 'num_devices',
 'num_countries', 'bba_17', 'num_urls', 'phone150', 'bba_18', 'bba_19',
 'phone55', 'phone33', 'bba_30', 'phone739', 'phone1030', 'my', 'bba_31',
 'bba_35', 'phone136', 'ca', 'bba_32', 'bba_33', 'cn', 'phone58', 'phone640',
 'za', 'in', 'phone15', 'phone996', 'ar', 'ru', 'ch', 'ec']
0.945600450453 0.0169479757094
0.960789561914 0.000756052446392

16. 
n_est = 2000
0.948751144697 0.015679560222
0.964327489414 0.00113223205984
['au', 'phone46', 'phone55', 'za', 'phone739', 'bba_4', 'num_devices', 'bba_5', 'my', 'de', 'bba_2', 'th', 'bba_6', 'bba_9', 'bba_3', 'num_bids', 'ca', 'bba_7', 'bba_8', 'bba_1', 'phone640', 'us', 'cn', 'num_urls', 'phone996', 'jp', 'phone136', 'num_countries', 'in', 'num_aucs', 'phone150', 'ch', 'num_ips', 'ar', 'bba_15', 'bba_14', 'phone33', 'bba_17', 'phone1030', 'phone58']


17
n_est = 2000
0.95383276018 0.0157806346745
0.962807988053 0.00133983098017
['au', 'phone46', 'phone143', 'phone739', 'phone55', 'za', 'sg', 'phone728', 'bba_4', 'bba_5', 'th', 'uk', 'num_devices', 'jp', 'bba_2', 'ca', 'ph', 'de', 'phone2287', 'it', 'my', 'bba_6', 'phone996', 'id', 'phone136', 'bba_3', 'phone21', 'phone110', 'bba_8', 'us', 'num_bids', 'bba_7', 'bba_1', 'phone28', 'bba_10', 'bba_9', 'phone63', 'phone640', 'cn', 'num_countries', 'bba_11', 'num_aucs', 'ch', 'fr', 'phone1026', 'num_urls', 'phone195', 'in', 'ar', 'phone150', 'phone469', 'phone219', 'phone144', 'bba_12', 'phone6', 'phone168', 'num_ips', 'phone25', 'phone2330', 'phone58', 'bn', 'bba_15', 'phone65', 'bba_14', 'phone33', 'phone1030', 'bba_13', 'bba_16', 'nl', 'phone22', 'ua', 'br', 'bba_17', 'phone3359', 'ru', 'no', 'phone90', 'bba_19', 'phone224', 'bba_21', 'bba_22', 'phone205', 'bh', 'bba_20', 'bba_18', 'bba_26', 'qa', 'bba_29', 'bba_24', 'bba_28', 'bba_27', 'bba_23', 'bba_25', 'bba_30', 'bba_32', 'bba_33', 'computers', 'lt', 'sa', 'bba_31', 'bba_35', 'mobile', 'phone15', 'sporting goods', 'lv', 'tw', 'jewelry', 'home goods', 'lu', 'kr', 'phone239', 'ec', 'office equipment', 'books and music', 'bf', 'auto parts', 'furniture', 'clothing', 'num_merchs']


18
n_est = 8000
max_features = 0.025
0.948030305337 0.017460442914
0.955229908369 0.000516114667192

119 features
