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


19. 

n_est = 8000
max_feat = 'auto'
0.956433766831 0.0136572167668
0.963114337586 0.000641687548135
['au', 'phone46', 'phone739', 'phone143', 'phone55', 'za', 'sg', '58', 'phone728', 'ca', 'th', 'num_devices', 'jp', 'id', 'uk', 'it', 'de', 'ph', 'phone2287', '8', 'phone996', '7', 'phone110', '1', 'my', '6', 'phone21', '0', 'bba_4', 'us', '27', 'cn', '3', 'bba_5', '4', 'phone28', '19', '18', '2', 'phone136', '9', '10', 'phone1026', 'num_countries', 'phone640', 'fr', 'phone219', 'bba_3', 'ch', 'bba_6', '11', 'bba_2', 'bba_8', 'bba_9', '12', 'bba_7', 'phone63', 'in', 'phone469', '21', 'num_bids', 'bba_10', 'ar', 'bba_1', '13', 'num_aucs', 'bba_11', '20', 'phone25', 'phone144', '5', 'bn', 'phone6', 'num_urls', 'phone2330', 'nl', 'phone150', 'num_ips', 'phone58', 'phone65', 'phone1030', 'ua', 'bba_12', 'br', 'computers', 'no', 'phone33', 'bba_14', 'phone195', 'phone3359', 'phone22', 'bba_15', 'bba_13', 'phone168', 'ru', 'bba_16', 'qa', 'bh', 'phone205', 'bba_17', 'bba_18', 'phone90', 'phone15', 'lt', 'bba_19', 'phone224', 'bba_27', 'bba_26', 'bba_25', 'bba_28', 'bba_21', 'bba_20', 'sa', 'bba_29', 'tw', 'bba_24', 'mobile', 'bba_23', 'bba_30', 'bba_22', 'bba_32', 'bba_31', 'bba_33', 'bba_35', 'sporting goods', 'jewelry', 'home goods', 'lv', 'lu', 'kr', 'phone239', 'ec', 'books and music', 'office equipment', 'bf', 'auto parts', 'furniture', 'clothing']

20.
n_est = 8000
max_feat = 0.025
0.952225725582 0.0204278890233
0.956947853692 0.00136960906959

n_feat = 138

20
n_est = 10,000
max_feat = 0.015
0.95054810477 0.019187434505
0.956040286808 0.000715517744663


22.
extratree: n_est = 10,000 max_feat = auto
baggingclassifier: n_est = 10, max_feat = 0.5
0.951130683016 0.0188514783235
0.958869857263 0.00118009854702

23.
extratree: n_est = 10,000 max_feat = 0.015
baggingclassifier: n_est = 10, max_feat = 0.75
0.949248376705 0.0205232286282
0.958061522021 0.00142994166689



24.
all features except bid_streaks
0.944137121986 0.0168897286188
0.958366590369 0.00113578948111

141 features
0.954206853482 0.0152990922129
0.962102956678 0.00171932441174

53 features, n_est=10000, max_feat=auto
0.957640781825 0.0152186955176
0.963115365626 0.000713891469549
['au', 'bba_4', 'num_bids', 'streak_4', 'int_1', 'bba_3', 'bba_5', 'bba_2', 'streak_3', 'int_2', 'bba_1', 'int_4', 'num_bids_sametime_diffauc', 'int_3', 'int_0', 'streak_1', 'streak_2', 'phone55', 'streak_0', 'num_urls', 'num_ips', 'num_devices', 'my', 'za', 'ca', 'num_aucs', 'phone739', 'phone150', 'num_countries', 'phone996', 'phone33', 'in', 'phone640', 'phone58', 'phone136', 'cn', 'ru', 'phone15', 'ar', 'num_bids_sametime_sameauc', 'ch', 'phone1030', 'ec', 'mobile', 'sporting goods', 'jewelry', 'computers', 'home goods', 'office equipment', 'books and music', 'auto parts', 'furniture', 'clothing']

0.955274285186 0.0162474707099
0.962406241415 0.000821079043834
class_weight=0.06
['au', 'phone55', 'phone739', 'za', 'ca', 'my', 'streak_4', 'bba_4', 'phone996', 'num_devices', 'streak_3', 'bba_5', 'streak_1', 'bba_3', 'bba_1', 'bba_2', 'num_bids', 'phone640', 'int_1', 'streak_2', 'num_bids_sametime_diffauc', 'cn', 'streak_0', 'int_4', 'num_countries', 'int_2', 'int_3', 'int_0', 'in', 'num_urls', 'num_aucs', 'phone136', 'ar', 'num_ips', 'ch', 'phone150', 'ru', 'phone58', 'phone33', 'phone15', 'phone1030', 'num_bids_sametime_sameauc', 'mobile', 'computers', 'sporting goods', 'jewelry', 'home goods', 'ec', 'books and music', 'office equipment', 'auto parts', 'furniture', 'clothing']


0.956120929511 0.0173861348224
0.964122910706 0.000320234312595

25.



28.
xgb
0.966210355521 0.0110413492381
max features (+url, +bids counts by periods)


29.
ET
n_est = 3000
0.955849939553 0.01446240735
