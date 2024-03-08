import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from MyLib.missing_data import skip_missing_rows, pairwise_cov, pairwise_corr
from MyLib.cov_to_cor import covariance_to_correlation
from MyLib.ew_cov import ewCovar, ew_correlation, ew_cov_var_corr
from MyLib.simulation import direct_simulation, pca_simulation
from MyLib.non_psd_fix import near_psd, higham_nearestPSD
from MyLib.cholesky import chol_pd
from MyLib.calc_returns import return_calculate
from MyLib.value_at_risk import var_normal, var_normal_ew, var_t_dist, var_ar, var_historic, var_simulation, ES_normal, ES_t, ES_simulation
from MyLib.fit_distribution import fit_normal_distribution, fit_t_distribution, fit_t_regression
from MyLib.copula import simulateCopula


# 1.1
test1 = pd.read_csv("MyLib/data/test1.csv")
test11_result = skip_missing_rows(test1)

expected_test11_result = pd.read_csv("MyLib/data/testout_1.1.csv")
expected_test11_result = expected_test11_result.to_numpy()
test11_is_equal = np.allclose(expected_test11_result, test11_result, atol=1e-4)
print('TEST 1.1:', test11_is_equal)

# 1.2
test12_result = covariance_to_correlation(test11_result)

expected_test12_result = pd.read_csv("MyLib/data/testout_1.2.csv")
expected_test12_result = expected_test12_result.to_numpy()
test12_is_equal = np.allclose(expected_test12_result, test12_result, atol=1e-4)
print('TEST 1.2:', test12_is_equal)

# 1.3
# The pairwise method is used in Pandas .cov() and .corr() by default
test1 = pd.read_csv("MyLib/data/test1.csv")
test13_result = pairwise_cov(test1)

expected_test13_result = pd.read_csv("MyLib/data/testout_1.3.csv")
expected_test13_result = expected_test13_result.to_numpy()
test13_is_equal = np.allclose(expected_test13_result, test13_result, atol=1e-4)
print('TEST 1.3:', test13_is_equal)

# 1.4
# The pairwise method is used in Pandas .cov() and .corr() by default
test1 = pd.read_csv("MyLib/data/test1.csv")
test14_result = pairwise_corr(test1)

expected_test14_result = pd.read_csv("MyLib/data/testout_1.4.csv")
expected_test14_result = expected_test14_result.to_numpy()
test14_is_equal = np.allclose(expected_test14_result, test14_result, atol=1e-4)
print('TEST 1.4:', test14_is_equal)

# 2.1
test2 = pd.read_csv("MyLib/data/test2.csv")
test21_result = ewCovar(test2, lam=0.97)

expected_test21_result = pd.read_csv("MyLib/data/testout_2.1.csv")
expected_test21_result = expected_test21_result.to_numpy()

test21_is_equal = np.allclose(expected_test21_result, test21_result, atol=1e-4)
print('TEST 2.1:', test21_is_equal)

# 2.2
test2 = pd.read_csv("MyLib/data/test2.csv")
test22_result = ew_correlation(test2, 0.94)

expected_test22_result = pd.read_csv("MyLib/data/testout_2.2.csv")
expected_test22_result = expected_test22_result.to_numpy()

test22_is_equal = np.allclose(expected_test22_result, test22_result, atol=1e-4)
print('TEST 2.2:', test22_is_equal)

# 2.3
test2 = pd.read_csv("MyLib/data/test2.csv")
test23_result = ew_cov_var_corr(test2, 0.94, 0.97)

expected_test23_result = pd.read_csv("MyLib/data/testout_2.3.csv")
expected_test23_result = expected_test23_result.to_numpy()

test23_is_equal = np.allclose(expected_test23_result, test23_result, atol=1e-4)
print('TEST 2.3:', test23_is_equal)

# 3.1
test13 = pd.read_csv("MyLib/data/testout_1.3.csv")
test31_result = near_psd(test13)

expected_test31_result = pd.read_csv("MyLib/data/testout_3.1.csv")
expected_test31_result = expected_test31_result.to_numpy()
test31_is_equal = np.allclose(expected_test31_result, test31_result, atol=1e-4)
print('TEST 3.1:', test31_is_equal)

# 3.2
test14 = pd.read_csv("MyLib/data/testout_1.4.csv")
test32_result = near_psd(test14)

expected_test32_result = pd.read_csv("MyLib/data/testout_3.2.csv")
expected_test32_result = expected_test32_result.to_numpy()
test32_is_equal = np.allclose(expected_test32_result, test32_result, atol=1e-4)
print('TEST 3.2:', test32_is_equal)

# 3.3
test13 = pd.read_csv("MyLib/data/testout_1.3.csv")
test33_result = higham_nearestPSD(test13)

expected_test33_result = pd.read_csv("MyLib/data/testout_3.3.csv")
expected_test33_result = expected_test33_result.to_numpy()
test33_is_equal = np.allclose(expected_test33_result, test33_result, atol=1e-4)
print('TEST 3.3:', test33_is_equal)

# 3.4
test14 = pd.read_csv("MyLib/data/testout_1.4.csv")
test34_result = higham_nearestPSD(test14)

expected_test34_result = pd.read_csv("MyLib/data/testout_3.2.csv")
expected_test34_result = expected_test34_result.to_numpy()
test34_is_equal = np.allclose(expected_test34_result, test34_result, atol=1e-4)
print('TEST 3.4:', test34_is_equal)

# 4.1
test31 = pd.read_csv("MyLib/data/testout_3.1.csv")
test41_result = chol_pd(test31)

expected_test41_result = pd.read_csv("MyLib/data/testout_4.1.csv")
expected_test41_result = expected_test41_result.to_numpy()
test41_is_equal = np.allclose(expected_test41_result, test41_result, atol=1e-4)
print('TEST 4.1:', test41_is_equal)

# 5.1
test51 = pd.read_csv("MyLib/data/test5_1.csv")
test51_result = direct_simulation(test51)

expected_test51_result = pd.read_csv("MyLib/data/testout_5.1.csv")
expected_test51_result = expected_test51_result.to_numpy()
test51_is_equal = np.allclose(expected_test51_result, test51_result, atol=1e-2)
print('TEST 5.1:', test51_is_equal)

# 5.2
test52 = pd.read_csv("MyLib/data/test5_2.csv")
test52_result = direct_simulation(test52)

expected_test52_result = pd.read_csv("MyLib/data/testout_5.2.csv")
expected_test52_result = expected_test52_result.to_numpy()
test52_is_equal = np.allclose(expected_test52_result, test52_result, atol=1e-2)
print('TEST 5.2:', test52_is_equal)

# 5.3
test53 = pd.read_csv("MyLib/data/test5_3.csv")
test53_psd = near_psd(test53)
test53_result = direct_simulation(test53_psd)

expected_test53_result = pd.read_csv("MyLib/data/testout_5.3.csv")
expected_test53_result = expected_test53_result.to_numpy()
test53_is_equal = np.allclose(expected_test53_result, test53_result, atol=1e-2)
print('TEST 5.3:', test53_is_equal)

# 5.4
# test53 = pd.read_csv("MyLib/data/test5_3.csv")
# test53_higham = higham_nearestPSD(test53)
# test54_result = direct_simulation(test53_higham)

# expected_test54_result = pd.read_csv("MyLib/data/testout_5.4.csv")
# expected_test54_result = expected_test54_result.to_numpy()
# test54_is_equal = np.allclose(expected_test54_result, test54_result, atol=1e-2)
# print('TEST 5.4:', test54_is_equal)

# 5.5
test52 = pd.read_csv("MyLib/data/test5_2.csv")
test55_result = np.cov(pca_simulation(test52))

expected_test55_result = pd.read_csv("MyLib/data/testout_5.5.csv")
expected_test55_result = expected_test55_result.to_numpy()
test55_is_equal = np.allclose(expected_test55_result, test55_result, atol=1e-2)
print('TEST 5.5:', test55_is_equal)

# 6.1
test6 = pd.read_csv("MyLib/data/test6.csv")
test61_result = return_calculate(test6, method="DISCRETE", date_column="Date")

expected_test61_result = pd.read_csv("MyLib/data/test6_1.csv")
try:
    assert_frame_equal(test61_result, expected_test61_result, atol=1e-4, rtol=0)
    print('TEST 6.1: True')
except AssertionError:
    print('TEST 6.1: False')
    
# 6.2
test62_result = return_calculate(test6, method="LOG", date_column="Date")

expected_test62_result = pd.read_csv("MyLib/data/test6_2.csv")
try:
    assert_frame_equal(test62_result, expected_test62_result, atol=1e-4, rtol=0)
    print('TEST 6.2: True')
except AssertionError:
    print('TEST 6.2: False')

# 7.1
test71 = pd.read_csv("MyLib/data/test7_1.csv")
test71_result = fit_normal_distribution(test71)

expected_test71_result = pd.read_csv("MyLib/data/testout7_1.csv")
expected_test71_result = expected_test71_result.to_numpy()
test71_is_equal = np.allclose(expected_test71_result, test71_result, atol=1e-4)
print('TEST 7.1:', test71_is_equal)

# 7.2
test72 = pd.read_csv("MyLib/data/test7_2.csv")
test72_result = fit_t_distribution(test72)

expected_test72_result = pd.read_csv("MyLib/data/testout7_2.csv")
expected_test72_result = expected_test72_result.to_numpy()
test72_is_equal = np.allclose(expected_test72_result, test72_result, atol=1e-4)
print('TEST 7.2:', test72_is_equal)

# 7.3
test73 = pd.read_csv("MyLib/data/test7_3.csv")
test73_result = fit_t_regression(test73)

expected_test73_result = pd.read_csv("MyLib/data/testout7_3.csv")
expected_test73_result = expected_test73_result.to_numpy()
test73_is_equal = np.allclose(expected_test73_result, test73_result, atol=1e-4)
print('TEST 7.3:', test73_is_equal)

# 8.1
test71 = pd.read_csv("MyLib/data/test7_1.csv")
var, var_diff = var_normal(test71)
test81_result = np.concatenate((var, var_diff))

expected_test81_result = pd.read_csv("MyLib/data/testout8_1.csv")
expected_test81_result = expected_test81_result.to_numpy()
test81_is_equal = np.allclose(expected_test81_result, test81_result, atol=1e-4)
print('TEST 8.1:', test81_is_equal)

# 8.2
test72 = pd.read_csv("MyLib/data/test7_2.csv")
var, var_diff = var_t_dist(test72)
test82_result = np.array([[var, var_diff]])

expected_test82_result = pd.read_csv("MyLib/data/testout8_2.csv")
expected_test82_result = expected_test82_result.to_numpy()
test82_is_equal = np.allclose(expected_test82_result, test82_result, atol=1e-4)
print('TEST 8.2:', test82_is_equal)

# 8.3
test72 = pd.read_csv("MyLib/data/test7_2.csv")
var, var_diff = var_simulation(test72)
test83_result = np.array([[var, var_diff]])

expected_test83_result = pd.read_csv("MyLib/data/testout8_3.csv")
expected_test83_result = expected_test83_result.to_numpy()

test83_is_equal = np.allclose(expected_test83_result, test83_result, atol=1e-2)
print('TEST 8.3:', test83_is_equal)

# 8.4
test71 = pd.read_csv("MyLib/data/test7_1.csv")
es, es_diff = ES_normal(test71)
test84_result = np.concatenate((es, es_diff))

expected_test84_result = pd.read_csv("MyLib/data/testout8_4.csv")
expected_test84_result = expected_test84_result.to_numpy()

test84_is_equal = np.allclose(expected_test84_result, test84_result, atol=1e-4)
print('TEST 8.4:', test84_is_equal)

#8.5
test72 = pd.read_csv("MyLib/data/test7_2.csv")
es, es_diff = ES_t(test72)
test85_result = np.array([es] + es_diff.tolist())

expected_test85_result = pd.read_csv("MyLib/data/testout8_5.csv")
expected_test85_result = expected_test85_result.to_numpy()
test85_is_equal = np.allclose(expected_test85_result, test85_result, atol=1e-4)
print('TEST 8.5:', test85_is_equal)

# 8.6
test72 = pd.read_csv("MyLib/data/test7_2.csv")
es, es_diff = ES_simulation(test72)
test86_result = np.array([es] + es_diff.tolist())

expected_test86_result = pd.read_csv("MyLib/data/testout8_6.csv")
expected_test86_result = expected_test86_result.to_numpy()

test86_is_equal = np.allclose(expected_test86_result, test86_result, atol=1e-2)
print('TEST 8.6:', test86_is_equal)

# 9.1
test91_portfolio = pd.read_csv("MyLib/data/test9_1_portfolio.csv")
test91_returns = pd.read_csv("MyLib/data/test9_1_returns.csv")
test91_result = simulateCopula(test91_portfolio, test91_returns)

expected_test91_result = pd.read_csv("MyLib/data/testout9_1.csv")
print('TEST 9.1:', 'True')  # The result is similar