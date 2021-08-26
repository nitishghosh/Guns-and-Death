import math
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats as st

def US_plot_scale():
    return 250000

def World_plot_scale():
    return 5000000

def US_murder_rate():
    return 5.35

def US_suicide_rate():
    return 16.1

# Not Used
def calc_weighted_median(col_a, col_w):
    data = np.array([])

    i = 0
    while i < len(col_a):
        j = 0
        while j < col_w[i]:
            data = np.append(data, col_a[i])
            j = j + 1
        i = i + 1
    data = np.sort(data)
    print(data)

    if len(data) % 2 == 0:
        median_index1 = len(data) / 2 - 1
        median_index2 = len(data) / 2
        median = (data[median_index1] + data[median_index2]) / 2
    else:
        median_index = (len(data) - 1) / 2

    median_index = int(median_index)
    median = data[median_index]
    return median # #

def calc_weighted_mean(col_a, col_w):
    i = 0
    j = 0
    total_weight = 0.0
    numerator_sum = 0.0

    while i < len(col_w):
        total_weight = total_weight + col_w[i]
        i = i + 1

    while j < len(col_a):
        numerator_sum = numerator_sum + col_w[j] * col_a[j]
        j = j + 1

    weighted_mean = numerator_sum / total_weight
    return weighted_mean

def calc_weighted_S_coeff(col_a, col_b, col_w):
    weighted_mean_a = calc_weighted_mean(col_a, col_w)
    weighted_mean_b = calc_weighted_mean(col_b, col_w)
    S_x_x = 0.0
    S_y_y = 0.0
    S_x_y = 0.0

    i = 0
    while i < len(col_w):
        S_x_x = S_x_x + col_w[i] * (col_a[i] - weighted_mean_a) * (col_a[i] - weighted_mean_a)
        S_y_y = S_y_y + col_w[i] * (col_b[i] - weighted_mean_b) * (col_b[i] - weighted_mean_b)
        S_x_y = S_x_y + col_w[i] * (col_a[i] - weighted_mean_a) * (col_b[i] - weighted_mean_b)
        i = i + 1

    S_coeff = np.array([S_x_x, S_y_y, S_x_y])
    return S_coeff

def calc_weighted_std(col_a, col_w):
    weighted_mean = calc_weighted_mean(col_a, col_w)
    i = 0
    numerator_sum = 0.0
    denominator_sum = 0.0

    while i < len(col_w) :
        numerator_sum = numerator_sum + col_w[i] * math.pow((col_a[i] - weighted_mean), 2)
        denominator_sum = denominator_sum + col_w[i]
        i = i + 1

    weighted_std = math.sqrt(numerator_sum / (denominator_sum * (len(col_w) / (len(col_w) - 1))))
    return weighted_std

def calc_weighted_covariance(col_a, col_b, col_w):
    i = 0
    j = 0
    total_weight = 0.0
    numerator_sum = 0.0

    weighted_mean1 = calc_weighted_mean(col_a, col_w)
    weighted_mean2 = calc_weighted_mean(col_b, col_w)

    while i < len(col_w):
        total_weight = total_weight + col_w[i]
        i = i + 1

    while j < len(col_w):
        numerator_sum = numerator_sum + col_w[j] * (col_a[j] - weighted_mean1) * (col_b[j] - weighted_mean2)
        j = j + 1

    weighted_covariance = numerator_sum / total_weight
    return weighted_covariance

def calc_weighted_correlation(col_a, col_b, col_w):
    cov_x_y = calc_weighted_covariance(col_a, col_b, col_w)
    cov_x_x = calc_weighted_covariance(col_a, col_a, col_w)
    cov_y_y = calc_weighted_covariance(col_b, col_b, col_w)
    weighted_correlation = (cov_x_y) / (math.sqrt(cov_x_x * cov_y_y))
    return weighted_correlation

def calc_p_value(correlation_test, correlation_est, n):
    z_score = (math.sqrt(n - 3) / 2) * math.log(((1 + correlation_est) * (1 - correlation_test)) / ((1 - correlation_est) * (1 + correlation_test)))
    p_value = 2 * (1 - st.norm.cdf(abs(z_score)))
    return p_value

def calc_weighted_reg_coeff(col_a, col_b, col_w):
    weighted_mean_a = calc_weighted_mean(col_a, col_w)
    weighted_mean_b = calc_weighted_mean(col_b, col_w)
    S_x_y = calc_weighted_S_coeff(col_a, col_b, col_w)[2]
    S_x_x = calc_weighted_S_coeff(col_a, col_b, col_w)[0]

    slope = S_x_y / S_x_x
    intercept = weighted_mean_b - slope * weighted_mean_a

    coeff = np.array([slope, intercept])
    return coeff

def US_output():
    US_Data = pd.read_csv('C:/Users/dg392003/Desktop/Gun Violence Project/US Data.csv')
    US_Data_Matrix = US_Data.values
    US_Data_Matrix = np.transpose(US_Data_Matrix)
    print(US_Data.to_string())

    gun_ownership_rate_US = US_Data_Matrix[1]
    num_gun_laws_US = US_Data_Matrix[2]
    murder_rate_US = US_Data_Matrix[3]
    suicide_rate_US = US_Data_Matrix[4]
    population_US = US_Data_Matrix[5]

    gun_ownership_rate_US_mean = calc_weighted_mean(gun_ownership_rate_US, population_US)
    gun_ownership_rate_US_std = calc_weighted_std(gun_ownership_rate_US, population_US)
    num_gun_laws_US_mean = calc_weighted_mean(num_gun_laws_US, population_US)
    num_gun_laws_US_std = calc_weighted_std(num_gun_laws_US, population_US)
    murder_rate_US_mean = calc_weighted_mean(murder_rate_US, population_US)
    murder_rate_US_std = calc_weighted_std(murder_rate_US, population_US)
    suicide_rate_US_mean = calc_weighted_mean(suicide_rate_US, population_US)
    suicide_rate_US_std = calc_weighted_std(suicide_rate_US, population_US)

    weighted_correlation_ownership_murder = calc_weighted_correlation(gun_ownership_rate_US, murder_rate_US, population_US)
    reg_coeff_ownership_murder = calc_weighted_reg_coeff(gun_ownership_rate_US, murder_rate_US, population_US)
    slope_ownership_murder = reg_coeff_ownership_murder[0]
    intercept_ownership_murder = reg_coeff_ownership_murder[1]
    p_value_ownership_murder = calc_p_value(0, weighted_correlation_ownership_murder, len(population_US))

    weighted_correlation_ownership_suicide = calc_weighted_correlation(gun_ownership_rate_US, suicide_rate_US, population_US)
    reg_coeff_ownership_suicide = calc_weighted_reg_coeff(gun_ownership_rate_US, suicide_rate_US, population_US)
    slope_ownership_suicide = reg_coeff_ownership_suicide[0]
    intercept_ownership_suicide = reg_coeff_ownership_suicide[1]
    p_value_ownership_suicide = calc_p_value(0, weighted_correlation_ownership_suicide, len(population_US))

    weighted_correlation_numlaws_murder = calc_weighted_correlation(num_gun_laws_US, murder_rate_US, population_US)
    reg_coeff_numlaws_murder = calc_weighted_reg_coeff(num_gun_laws_US, murder_rate_US, population_US)
    slope_numlaws_murder = reg_coeff_numlaws_murder[0]
    intercept_numlaws_murder = reg_coeff_numlaws_murder[1]
    p_value_numlaws_murder = calc_p_value(0, weighted_correlation_numlaws_murder, len(population_US))

    weighted_correlation_numlaws_suicide = calc_weighted_correlation(num_gun_laws_US, suicide_rate_US, population_US)
    reg_coeff_numlaws_suicide = calc_weighted_reg_coeff(num_gun_laws_US, suicide_rate_US, population_US)
    slope_numlaws_suicide = reg_coeff_numlaws_suicide[0]
    intercept_numlaws_suicide = reg_coeff_numlaws_suicide[1]
    p_value_numlaws_suicide = calc_p_value(0, weighted_correlation_numlaws_suicide, len(population_US))

    gun_ownership_rate_US_zscore = np.zeros(len(population_US))
    gun_ownership_rate_US_percentile = np.zeros(len(population_US))
    num_gun_laws_US_zscore = np.zeros(len(population_US))
    num_gun_laws_US_percentile = np.zeros(len(population_US))

    plot1_sizes = population_US.astype('f')  # array sizing only works for floats
    plot2_sizes = population_US.astype('f')
    plot3_sizes = population_US.astype('f')
    plot4_sizes = population_US.astype('f')

    i = 0
    while i < len(gun_ownership_rate_US):
        gun_ownership_rate_US_zscore[i] = (gun_ownership_rate_US[i] - gun_ownership_rate_US_mean) / gun_ownership_rate_US_std
        gun_ownership_rate_US_percentile[i] = 100 * st.norm.cdf(gun_ownership_rate_US_zscore[i])
        num_gun_laws_US_zscore[i] = (num_gun_laws_US[i] - num_gun_laws_US_mean) / num_gun_laws_US_std
        num_gun_laws_US_percentile[i] = 100 * st.norm.cdf(num_gun_laws_US_zscore[i])
        plot1_sizes[i] = plot1_sizes[i] / US_plot_scale()
        plot2_sizes[i] = plot2_sizes[i] / US_plot_scale()
        plot3_sizes[i] = plot3_sizes[i] / US_plot_scale()
        plot4_sizes[i] = plot4_sizes[i] / US_plot_scale()
        i = i + 1

    print("------------------------------------------------------------SUMMARY STATISTICS-------------------------------------------------------")
    print("The mean US estimated gun ownership is " + str(gun_ownership_rate_US_mean) + ".")
    print("The standard deviation of US estimated gun ownership rate is " + str(gun_ownership_rate_US_std) + ".")
    print("The mean US number of gun laws is " + str(num_gun_laws_US_mean) + ".")
    print("The standard deviation of US number of gun laws is " + str(num_gun_laws_US_std) + ".")
    print("The mean US murder rate per 100,000 is " + str(murder_rate_US_mean) + ".")
    print("The standard deviation of US murder rate per 100,000 is " + str(murder_rate_US_std) + ".")
    print("The mean US suicide rate per 100,000 is " + str(suicide_rate_US_mean) + ".")
    print("The standard deviation of US suicide rate per 100,000 is " + str(suicide_rate_US_std) + ".")
    print("------------------------------------------------------------CORRELATION ANALYSIS-----------------------------------------------------")
    print("The weighted correlation between gun ownership and murder rate within the US is " + str(weighted_correlation_ownership_murder) + ".")
    print("This yields an R squared value of " + str(math.pow(weighted_correlation_ownership_murder, 2)) + ".")
    print("The slope is " + str(slope_ownership_murder) + ".")
    print("The p value is " + str(p_value_ownership_murder) + ".")
    print("The weighted correlation between gun ownership and suicide rate within the US is " + str(weighted_correlation_ownership_suicide) + ".")
    print("This yields an R squared value of " + str(math.pow(weighted_correlation_ownership_suicide, 2)) + ".")
    print("The slope is " + str(slope_ownership_suicide) + ".")
    print("The p value is " + str(p_value_ownership_suicide) + ".")
    print("The weighted correlation between the number of gun laws and murder rate within the US is " + str(weighted_correlation_numlaws_murder) + ".")
    print("This yields an R squared value of " + str(math.pow(weighted_correlation_numlaws_murder, 2)) + ".")
    print("The slope is " + str(slope_numlaws_murder) + ".")
    print("The p value is " + str(p_value_numlaws_murder) + ".")
    print("The weighted correlation between the number of gun laws and the suicide rate within the US is " + str(weighted_correlation_numlaws_suicide) + ".")
    print("This yields an R squared value of " + str(math.pow(weighted_correlation_numlaws_suicide, 2)) + ".")
    print("The slope is " + str(slope_numlaws_suicide) + ".")
    print("The p value is " + str(p_value_numlaws_suicide) + ".")

    plt.scatter(gun_ownership_rate_US, murder_rate_US, s = plot1_sizes, c = gun_ownership_rate_US_percentile, cmap = 'RdBu_r')
    plt.title("Murder Rate vs Gun Ownership Rate in the US")
    plt.xlabel("Estimated Gun Ownership Rate")
    plt.ylabel("Murder Rate")
    plt.axhline(y = murder_rate_US_mean, color = 'k', label = "Mean Murder Rate")
    plt.axvline(x = gun_ownership_rate_US_mean, color = 'k', label = "Mean Gun Ownership Rate")
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Gun Ownership Rate Percentile")
    plt.legend(prop = {'size' : 7})
    plt.plot(np.array([0, 0.6]), np.array([intercept_ownership_murder, slope_ownership_murder * 0.6 + intercept_ownership_murder]), color = 'k')
    plt.show()

    plt.scatter(gun_ownership_rate_US, suicide_rate_US, s = plot1_sizes, c = gun_ownership_rate_US_percentile, cmap = 'RdBu_r')
    plt.title("Suicide Rate vs Gun Ownership Rate in the US")
    plt.xlabel("Estimated Gun Ownership Rate")
    plt.ylabel("Suicide Rate")
    plt.axhline(y = suicide_rate_US_mean, color = 'k', label = "Mean Suicide Rate")
    plt.axvline(x = gun_ownership_rate_US_mean, color = 'k', label = "Mean Gun Ownership Rate")
    cbar2 = plt.colorbar()
    cbar2.ax.set_ylabel("Gun Ownership Rate Percentile")
    plt.legend(prop = {'size' : 7})
    plt.plot(np.array([0, 0.6]), np.array([intercept_ownership_suicide, slope_ownership_suicide * 0.6 + intercept_ownership_suicide]), color = 'k')
    plt.show()

    plt.scatter(num_gun_laws_US, murder_rate_US, s = plot1_sizes, c = num_gun_laws_US_percentile, cmap = "RdBu")
    plt.title("Murder Rate vs Number of Gun Laws in the US")
    plt.xlabel("Number of Gun Laws")
    plt.ylabel("Murder Rate")
    plt.axhline(y = murder_rate_US_mean, color = 'k', label = "Mean Murder Rate")
    plt.axvline(x = num_gun_laws_US_mean, color = 'k', label = "Mean Number of Gun Laws")
    cbar3 = plt.colorbar()
    cbar3.ax.set_ylabel("Number of Gun Laws Percentile")
    plt.legend(prop = {'size' : 7})
    plt.plot(np.array([0, 100]), np.array([intercept_numlaws_murder, slope_numlaws_murder * 100 + intercept_numlaws_murder]), color = 'k')
    plt.show()

    plt.scatter(num_gun_laws_US, suicide_rate_US, s = plot1_sizes, c = num_gun_laws_US_percentile, cmap = "RdBu")
    plt.title("Suicide Rate vs Number of Gun Laws in the US")
    plt.xlabel("Number of Gun Laws")
    plt.ylabel("Suicide Rate")
    plt.axhline(y = suicide_rate_US_mean, color = 'k', label = "Mean Suicide Rate")
    plt.axvline(x = num_gun_laws_US_mean, color = 'k', label = "Mean Number of Gun Laws")
    cbar4 = plt.colorbar()
    cbar4.ax.set_ylabel("Number of Gun Laws Percentile")
    plt.legend(prop = {'size' : 7})
    plt.plot(np.array([0, 100]), np.array([intercept_numlaws_suicide, slope_numlaws_suicide * 100 + intercept_numlaws_suicide]), color = 'k')
    plt.show()

def DevelopedCountry_output():
    DevelopedCountry_Data = pd.read_csv('C:/Users/dg392003/Desktop/Gun Violence Project/Developed Country Data.csv')
    DevelopedCountry_Data_Matrix = DevelopedCountry_Data.values
    DevelopedCountry_Data_Matrix = np.transpose(DevelopedCountry_Data_Matrix)
    print(DevelopedCountry_Data.to_string())

    gun_prevalence_developed_country = DevelopedCountry_Data_Matrix[1]
    gun_law_strictness_developed_country = DevelopedCountry_Data_Matrix[2]
    murder_rate_developed_country = DevelopedCountry_Data_Matrix[3]
    suicide_rate_developed_country = DevelopedCountry_Data_Matrix[4]
    population_developed_country = DevelopedCountry_Data_Matrix[5]

    gun_prevalence_developed_country_mean = calc_weighted_mean(gun_prevalence_developed_country, population_developed_country)
    gun_prevalence_developed_country_std = calc_weighted_std(gun_prevalence_developed_country, population_developed_country)
    murder_rate_developed_country_mean = calc_weighted_mean(murder_rate_developed_country, population_developed_country)
    murder_rate_developed_country_std = calc_weighted_std(murder_rate_developed_country, population_developed_country)
    suicide_rate_developed_country_mean = calc_weighted_mean(suicide_rate_developed_country, population_developed_country)
    suicide_rate_developed_country_std = calc_weighted_std(suicide_rate_developed_country, population_developed_country)

    gun_prevalance_developed_country_zscore = np.zeros(len(population_developed_country))
    gun_prevalance_developed_country_percentile = np.zeros(len(population_developed_country))

    plot1_sizes = population_developed_country.astype('f')
    plot2_sizes = population_developed_country.astype('f')

    i = 0
    while i < len(population_developed_country):
        gun_prevalance_developed_country_zscore[i] = (gun_prevalence_developed_country[i] - gun_prevalence_developed_country_mean) / gun_prevalence_developed_country_std
        gun_prevalance_developed_country_percentile[i] = 100 * st.norm.cdf(gun_prevalance_developed_country_zscore[i])
        plot1_sizes[i] = plot1_sizes[i] / US_plot_scale() # The US scale works here too
        plot2_sizes[i] = plot2_sizes[i] / US_plot_scale()
        i = i + 1

    weighted_correlation_prevalence_murder = calc_weighted_correlation(gun_prevalence_developed_country, murder_rate_developed_country, population_developed_country)
    reg_coeff_prevalence_murder = calc_weighted_reg_coeff(gun_prevalence_developed_country, murder_rate_developed_country, population_developed_country)
    slope_prevalence_murder = reg_coeff_prevalence_murder[0]
    intercept_prevalence_murder = reg_coeff_prevalence_murder[1]
    p_value_prevalence_murder = calc_p_value(0, weighted_correlation_prevalence_murder, len(population_developed_country))

    weighted_correlation_prevalence_suicide = calc_weighted_correlation(gun_prevalence_developed_country, suicide_rate_developed_country, population_developed_country)
    reg_coeff_prevalence_suicide = calc_weighted_reg_coeff(gun_prevalence_developed_country, suicide_rate_developed_country, population_developed_country)
    slope_prevalence_suicide = reg_coeff_prevalence_suicide[0]
    intercept_prevalance_suicide = reg_coeff_prevalence_suicide[1]
    p_value_prevalence_suicide = calc_p_value(0, weighted_correlation_prevalence_suicide, len(population_developed_country))

    print("------------------------------------------------------------SUMMARY STATISTICS-------------------------------------------------------")
    print("The mean gun prevalence is " + str(gun_prevalence_developed_country_mean) + ".")
    print("The standard deviation of gun prevalence is " + str(gun_prevalence_developed_country_std) + ".")
    print("The mean murder rate is " + str(murder_rate_developed_country_mean) + ".")
    print("The standard deviation of murder rate is " + str(murder_rate_developed_country_std) + ".")
    print("The mean suicide rate is " + str(suicide_rate_developed_country_mean) + ".")
    print("The standard deviation of suicide rate is " + str(suicide_rate_developed_country_std) + ".")

    print("------------------------------------------------------------CORRELATION ANALYSIS-----------------------------------------------------")
    print("The weighted correlation between gun prevalence and murder rate within the developed world is " + str(weighted_correlation_prevalence_murder) + ".")
    print("This yields an R squared value of " + str(math.pow(weighted_correlation_prevalence_murder, 2)) + ".")
    print("The slope is " + str(slope_prevalence_murder) + ".")
    print("The p value is " + str(p_value_prevalence_murder) + ".")
    print("The weighted correlation between gun prevalence and suicide rate within the developed world is " + str(weighted_correlation_prevalence_suicide) + ".")
    print("This yields an R squared value of " + str(math.pow(weighted_correlation_prevalence_suicide, 2)) + ".")
    print("The slope is " + str(slope_prevalence_suicide) + ".")
    print("The p value is " + str(p_value_prevalence_suicide) + ".")

    plt.scatter(gun_prevalence_developed_country, murder_rate_developed_country, s = plot1_sizes, c = gun_prevalance_developed_country_percentile, cmap = 'RdBu_r')
    plt.title("Murder Rate vs Gun Prevalence Rate in the Developed World")
    plt.xlabel("Gun Prevalence Rate")
    plt.ylabel("Murder Rate")
    plt.axhline(y = murder_rate_developed_country_mean, color = 'k', label = "Mean Murder Rate")
    plt.axvline(x = gun_prevalence_developed_country_mean, color = 'k', label = "Mean Gun Prevalence Rate")
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Gun Prevalence Rate Percentile")
    plt.legend(prop = {'size': 7})
    plt.plot(np.array([0, 35]), np.array([intercept_prevalence_murder, slope_prevalence_murder * 35 + intercept_prevalence_murder]), color='k')
    plt.show()

    plt.scatter(gun_prevalence_developed_country, suicide_rate_developed_country, s = plot2_sizes, c = gun_prevalance_developed_country_percentile, cmap = "RdBu_r")
    plt.title("Suicide Rate vs Gun Prevalence Rate in the Developed World")
    plt.xlabel("Gun Prevalence Rate")
    plt.ylabel("Suicide Rate")
    plt.axhline(y = suicide_rate_developed_country_mean, color = 'k', label = "Mean Suicide Rate")
    plt.axvline(x = gun_prevalence_developed_country_mean, color = 'k', label = "Mean Gun Prevalence Rate")
    cbar2 = plt.colorbar()
    cbar2.ax.set_ylabel("Gun Prevalence Rate Percentile")
    plt.legend(prop = {'size': 7})
    plt.plot(np.array([0, 35]), np.array([intercept_prevalance_suicide, slope_prevalence_suicide * 35 + intercept_prevalance_suicide]), color = 'k')
    plt.show()

    DevelopedCountry_Data_Matrix = np.transpose(DevelopedCountry_Data_Matrix)
    gun_law_strictness_murder_A = np.array([])
    gun_law_strictness_suicide_A = np.array([])
    weights_gun_law_strictness_A = np.array([])
    gun_law_strictness_murder_B = np.array([])
    gun_law_strictness_suicide_B = np.array([])
    weights_gun_law_strictness_B = np.array([])
    gun_law_strictness_murder_C = np.array([])
    gun_law_strictness_suicide_C = np.array([])
    weights_gun_law_strictness_C = np.array([])

    i = 0
    while i < len(population_developed_country):
        if DevelopedCountry_Data_Matrix[i][2] == 'A':
            gun_law_strictness_murder_A = np.append(gun_law_strictness_murder_A, DevelopedCountry_Data_Matrix[i][3])
            gun_law_strictness_suicide_A = np.append(gun_law_strictness_suicide_A, DevelopedCountry_Data_Matrix[i][4])
            weights_gun_law_strictness_A = np.append(weights_gun_law_strictness_A, DevelopedCountry_Data_Matrix[i][5])
        elif DevelopedCountry_Data_Matrix[i][2] == 'B':
            gun_law_strictness_murder_B = np.append(gun_law_strictness_murder_B, DevelopedCountry_Data_Matrix[i][3])
            gun_law_strictness_suicide_B = np.append(gun_law_strictness_suicide_B, DevelopedCountry_Data_Matrix[i][4])
            weights_gun_law_strictness_B = np.append(weights_gun_law_strictness_B, DevelopedCountry_Data_Matrix[i][5])
        elif DevelopedCountry_Data_Matrix[i][2] == 'C':
            gun_law_strictness_murder_C = np.append(gun_law_strictness_murder_C, DevelopedCountry_Data_Matrix[i][3])
            gun_law_strictness_suicide_C = np.append(gun_law_strictness_suicide_C, DevelopedCountry_Data_Matrix[i][4])
            weights_gun_law_strictness_C = np.append(weights_gun_law_strictness_C, DevelopedCountry_Data_Matrix[i][5])
        i = i + 1

    weighted_mean_murder_A = calc_weighted_mean(gun_law_strictness_murder_A, weights_gun_law_strictness_A)
    weighted_std_murder_A = calc_weighted_std(gun_law_strictness_murder_A, weights_gun_law_strictness_A)
    weighted_mean_suicide_A = calc_weighted_mean(gun_law_strictness_suicide_A, weights_gun_law_strictness_A)
    weighted_std_suicide_A = calc_weighted_std(gun_law_strictness_suicide_A, weights_gun_law_strictness_A)
    weighted_mean_murder_B = calc_weighted_mean(gun_law_strictness_murder_B, weights_gun_law_strictness_B)
    weighted_std_murder_B = calc_weighted_std(gun_law_strictness_murder_B, weights_gun_law_strictness_B)
    weighted_mean_suicide_B = calc_weighted_mean(gun_law_strictness_suicide_B, weights_gun_law_strictness_B)
    weighted_std_suicide_B = calc_weighted_std(gun_law_strictness_suicide_B, weights_gun_law_strictness_B)
    weighted_mean_murder_C = calc_weighted_mean(gun_law_strictness_murder_C, weights_gun_law_strictness_C)
    weighted_std_murder_C = calc_weighted_std(gun_law_strictness_murder_C, weights_gun_law_strictness_C)
    weighted_mean_suicide_C = calc_weighted_mean(gun_law_strictness_suicide_C, weights_gun_law_strictness_C)
    weighted_std_suicide_C = calc_weighted_std(gun_law_strictness_suicide_C, weights_gun_law_strictness_C)

    t_score_murder_a_b = (weighted_mean_murder_A - weighted_mean_murder_B) / math.sqrt((math.pow(weighted_std_murder_A, 2) / len(gun_law_strictness_murder_A)) + (math.pow(weighted_std_murder_B, 2) / len(gun_law_strictness_murder_B)))
    p_value_murder_a_b = 2 * (1 - st.t.cdf(abs(t_score_murder_a_b), len(gun_law_strictness_murder_A) + len(gun_law_strictness_murder_B) - 2))
    t_score_suicide_a_b = (weighted_mean_suicide_A - weighted_mean_suicide_B) / math.sqrt((math.pow(weighted_std_suicide_A, 2) / len(gun_law_strictness_suicide_A)) + (math.pow(weighted_std_suicide_B, 2) / len(gun_law_strictness_suicide_B)))
    p_value_suicide_a_b = 2 * (1 - st.t.cdf(abs(t_score_suicide_a_b), len(gun_law_strictness_suicide_A) + len(gun_law_strictness_suicide_B) - 2))
    t_score_murder_a_c = (weighted_mean_murder_A - weighted_mean_murder_C) / math.sqrt((math.pow(weighted_std_murder_A, 2) / len(gun_law_strictness_murder_A)) + (math.pow(weighted_std_murder_C, 2) / len(gun_law_strictness_murder_C)))
    p_value_murder_a_c = 2 * (1 - st.t.cdf(abs(t_score_murder_a_c), len(gun_law_strictness_murder_A) + len(gun_law_strictness_murder_C) - 2))
    t_score_suicide_a_c = (weighted_mean_suicide_A - weighted_mean_suicide_C) / math.sqrt((math.pow(weighted_std_suicide_A, 2) / len(gun_law_strictness_suicide_A)) + (math.pow(weighted_std_suicide_C, 2) / len(gun_law_strictness_suicide_C)))
    p_value_suicide_a_c = 2 * (1 - st.t.cdf(abs(t_score_suicide_a_c), len(gun_law_strictness_suicide_A) + len(gun_law_strictness_suicide_C) - 2))
    t_score_murder_b_c = (weighted_mean_murder_B - weighted_mean_murder_C) / math.sqrt((math.pow(weighted_std_murder_B, 2) / len(gun_law_strictness_murder_B)) + (math.pow(weighted_std_murder_C, 2) / len(gun_law_strictness_murder_C)))
    p_value_murder_b_c = 2 * (1 - st.t.cdf(abs(t_score_murder_b_c), len(gun_law_strictness_murder_B) + len(gun_law_strictness_murder_C) - 2))
    t_score_suicide_b_c = (weighted_mean_suicide_B - weighted_mean_suicide_C) / math.sqrt((math.pow(weighted_std_suicide_B, 2) / len(gun_law_strictness_suicide_C)) + (math.pow(weighted_std_suicide_C, 2) / len(gun_law_strictness_suicide_C)))
    p_value_suicide_b_c = 2 * (1 - st.t.cdf(abs(t_score_suicide_b_c), len(gun_law_strictness_suicide_B) + len(gun_law_strictness_suicide_C) - 2))

    print("------------------------------------------------------------T Test---------------------------------------------------------------------")
    print("The mean murder rate for the lax gun control category is " + str(weighted_mean_murder_A) + ".")
    print("The standard deviation murder rate for the lax gun control category is " + str(weighted_std_murder_A) + ".")
    print("The mean murder rate for the restrictive gun control category is " + str(weighted_mean_murder_B) + ".")
    print("The standard deviation murder rate for the restrictive gun control category is " + str(weighted_std_murder_B) + ".")
    print("The mean murder rate for the prohibited gun control category is " + str(weighted_mean_murder_C) + ".")
    print("The standard deviation murder rate for the prohibited gun control category is " + str(weighted_std_murder_C) + ".")
    print("The p value for murder between lax and restrictive gun control is " + str(p_value_murder_a_b) + ".")
    print("The p value for murder between lax and prohibited gun control is " + str(p_value_murder_a_c) + ".")
    print("The p value for murder between restrictive and prohibited gun control is " + str(p_value_murder_b_c) + ".")
    print("The mean suicide rate for the lax gun control category is " + str(weighted_mean_suicide_A) + ".")
    print("The standard deviation suicide rate for the lax gun control category is " + str(weighted_std_suicide_A) + ".")
    print("The mean suicide rate for the restrictive gun control category is " + str(weighted_mean_suicide_B) + ".")
    print("The standard deviation suicide rate for the restrictive gun control category is " + str(weighted_std_suicide_B) + ".")
    print("The mean suicide rate for the prohibited gun control category is " + str(weighted_mean_suicide_C) + ".")
    print("The standard deviation suicide rate for the prohibited gun control category is " + str(weighted_std_suicide_C) + ".")
    print("The p value for suicide between lax and restrictive gun control is " + str(p_value_suicide_a_b) + ".")
    print("The p value for suicide between lax and prohibited gun control is " + str(p_value_suicide_a_c) + ".")
    print("The p value for suicide between restrictive and prohibited gun control is " + str(p_value_suicide_b_c) + ".")

def World_output():
    World_Data = pd.read_csv('C:/Users/dg392003/Desktop/Gun Violence Project/World Data.csv')
    World_Data_Matrix = World_Data.values
    World_Data_Matrix = np.transpose(World_Data_Matrix)
    print(World_Data.to_string())

    gun_prevalence_rate_world = World_Data_Matrix[1]
    gun_law_strictness_world = World_Data_Matrix[2]
    murder_rate_world = World_Data_Matrix[3]
    suicide_rate_world = World_Data_Matrix[4]
    population_world = World_Data_Matrix[5]

    gun_prevalence_rate_world_mean = calc_weighted_mean(gun_prevalence_rate_world, population_world)
    gun_prevalence_rate_world_std = calc_weighted_std(gun_prevalence_rate_world, population_world)
    murder_rate_world_mean = calc_weighted_mean(murder_rate_world, population_world)
    murder_rate_world_std = calc_weighted_std(murder_rate_world, population_world)
    suicide_rate_world_mean = calc_weighted_mean(suicide_rate_world, population_world)
    suicide_rate_world_std = calc_weighted_std(suicide_rate_world, population_world)

    weighted_correlation_prevalence_murder = calc_weighted_correlation(gun_prevalence_rate_world, murder_rate_world, population_world)
    reg_coeff_prevalence_murder = calc_weighted_reg_coeff(gun_prevalence_rate_world, murder_rate_world, population_world)
    slope_prevalence_murder = reg_coeff_prevalence_murder[0]
    intercept_prevalence_murder = reg_coeff_prevalence_murder[1]
    p_value_prevalence_murder = calc_p_value(0, weighted_correlation_prevalence_murder, len(population_world))

    weighted_correlation_prevalence_suicide = calc_weighted_correlation(gun_prevalence_rate_world, suicide_rate_world, population_world)
    reg_coeff_prevalence_suicide = calc_weighted_reg_coeff(gun_prevalence_rate_world, suicide_rate_world, population_world)
    slope_prevalence_suicide = reg_coeff_prevalence_suicide[0]
    intercept_prevalence_suicide = reg_coeff_prevalence_suicide[1]
    p_value_prevalence_suicide = calc_p_value(0, weighted_correlation_prevalence_suicide, len(population_world))

    gun_prevalence_rate_world_zscore = np.zeros(len(population_world))
    gun_prevalence_rate_world_percentile = np.zeros(len(population_world))

    plot1_sizes = population_world.astype('f')  # array sizing only works for floats
    plot2_sizes = population_world.astype('f')

    i = 0
    while i < len(gun_prevalence_rate_world):
        gun_prevalence_rate_world_zscore[i] = (gun_prevalence_rate_world[i] - gun_prevalence_rate_world_mean) / gun_prevalence_rate_world_std
        gun_prevalence_rate_world_percentile[i] = 100 * st.norm.cdf(gun_prevalence_rate_world_zscore[i])
        plot1_sizes[i] = plot1_sizes[i] / World_plot_scale()
        plot2_sizes[i] = plot2_sizes[i] / World_plot_scale()
        i = i + 1

    print("------------------------------------------------------------SUMMARY STATISTICS-------------------------------------------------------")
    print("The mean world estimated gun prevalence is " + str(gun_prevalence_rate_world_mean) + ".")
    print("The standard deviation of world estimated gun prevalence is " + str(gun_prevalence_rate_world_std) + ".")
    print("The mean world murder rate per 100,000 is " + str(murder_rate_world_mean) + ".")
    print("The standard deviation of world murder rate per 100,000 is " + str(murder_rate_world_std) + ".")
    print("The mean world suicide rate per 100,000 is " + str(suicide_rate_world_mean) + ".")
    print("The standard deviation of world suicide rate per 100,000 is " + str(suicide_rate_world_std) + ".")
    print("------------------------------------------------------------CORRELATION ANALYSIS-----------------------------------------------------")
    print("The weighted correlation between gun prevalence and murder rate within the world is " + str(weighted_correlation_prevalence_murder) + ".")
    print("This yields an R squared value of " + str(math.pow(weighted_correlation_prevalence_murder, 2)) + ".")
    print("The slope is " + str(slope_prevalence_murder) + ".")
    print("The p value is " + str(p_value_prevalence_murder) + ".")
    print("The weighted correlation between gun prevalence and suicide rate within the world is " + str(weighted_correlation_prevalence_suicide))
    print("This yields an R squared value of " + str(math.pow(weighted_correlation_prevalence_suicide, 2)) + ".")
    print("The slope is " + str(slope_prevalence_suicide) + ".")
    print("The p value is " + str(p_value_prevalence_suicide) + ".")

    plt.scatter(gun_prevalence_rate_world, murder_rate_world, s = plot1_sizes, c = gun_prevalence_rate_world_percentile, cmap = 'RdBu_r')
    plt.title("Murder Rate vs Gun Prevalence Rate in the World (All)")
    plt.xlabel("Estimated Gun Prevalence Rate")
    plt.ylabel("Murder Rate")
    plt.axhline(y = murder_rate_world_mean, color = 'k', label = "Mean Murder Rate")
    plt.axvline(x = gun_prevalence_rate_world_mean, color = 'k', label = "Mean Gun Prevalence Rate")
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Gun Prevalence Rate Percentile")
    plt.legend(prop={'size': 7})
    plt.plot(np.array([0, 120]), np.array([intercept_prevalence_murder, slope_prevalence_murder * 120 + intercept_prevalence_murder]), color = 'k')
    plt.show()

    plt.scatter(gun_prevalence_rate_world, suicide_rate_world, s = plot1_sizes, c = gun_prevalence_rate_world_percentile, cmap = 'RdBu_r')
    plt.title("Suicide Rate vs Gun Prevalence Rate in the World (All)")
    plt.xlabel("Estimated Gun Prevalence Rate")
    plt.ylabel("Suicide Rate")
    plt.axhline(y = suicide_rate_world_mean, color = 'k', label = "Mean Suicide Rate")
    plt.axvline(x = gun_prevalence_rate_world_mean, color = 'k', label = "Mean Gun Ownership Rate")
    cbar2 = plt.colorbar()
    cbar2.ax.set_ylabel("Gun Prevalence Rate Percentile")
    plt.legend(prop={'size': 7})
    plt.plot(np.array([0, 120]), np.array([intercept_prevalence_suicide, slope_prevalence_suicide * 120 + intercept_prevalence_suicide]), color='k')
    plt.show()

    World_Data_Matrix = np.transpose(World_Data_Matrix)
    gun_law_strictness_murder_A = np.array([])
    gun_law_strictness_suicide_A = np.array([])
    weights_gun_law_strictness_A = np.array([])
    gun_law_strictness_murder_B = np.array([])
    gun_law_strictness_suicide_B = np.array([])
    weights_gun_law_strictness_B = np.array([])
    gun_law_strictness_murder_C = np.array([])
    gun_law_strictness_suicide_C = np.array([])
    weights_gun_law_strictness_C = np.array([])

    murder_less_than_US = np.array([])
    weights_murder_less_than_US = np.array([])
    murder_greater_than_US = np.array([])
    weights_murder_greater_than_US = np.array([])
    suicide_less_than_US = np.array([])
    weights_suicide_less_than_US = np.array([])
    suicide_greater_than_US = np.array([])
    weights_suicide_greater_than_US = np.array([])

    i = 0
    while i < len(population_world):
        if World_Data_Matrix[i][2] == 'A':
            gun_law_strictness_murder_A = np.append(gun_law_strictness_murder_A, World_Data_Matrix[i][3])
            gun_law_strictness_suicide_A = np.append(gun_law_strictness_suicide_A, World_Data_Matrix[i][4])
            weights_gun_law_strictness_A = np.append(weights_gun_law_strictness_A, World_Data_Matrix[i][5])
        elif World_Data_Matrix[i][2] == 'B':
            gun_law_strictness_murder_B = np.append(gun_law_strictness_murder_B, World_Data_Matrix[i][3])
            gun_law_strictness_suicide_B = np.append(gun_law_strictness_suicide_B, World_Data_Matrix[i][4])
            weights_gun_law_strictness_B = np.append(weights_gun_law_strictness_B, World_Data_Matrix[i][5])
        elif World_Data_Matrix[i][2] == 'C':
            gun_law_strictness_murder_C = np.append(gun_law_strictness_murder_C, World_Data_Matrix[i][3])
            gun_law_strictness_suicide_C = np.append(gun_law_strictness_suicide_C, World_Data_Matrix[i][4])
            weights_gun_law_strictness_C = np.append(weights_gun_law_strictness_C, World_Data_Matrix[i][5])
        if World_Data_Matrix[i][3] < US_murder_rate():
            murder_less_than_US = np.append(murder_less_than_US, World_Data_Matrix[i][1])
            weights_murder_less_than_US = np.append(weights_murder_less_than_US, World_Data_Matrix[i][5])
        elif World_Data_Matrix[i][3] > US_murder_rate():
            murder_greater_than_US = np.append(murder_greater_than_US, World_Data_Matrix[i][1])
            weights_murder_greater_than_US = np.append(weights_murder_greater_than_US, World_Data_Matrix[i][5])
        if World_Data_Matrix[i][4] < US_suicide_rate():
            suicide_less_than_US = np.append(suicide_less_than_US, World_Data_Matrix[i][1])
            weights_suicide_less_than_US = np.append(weights_suicide_less_than_US, World_Data_Matrix[i][5])
        elif World_Data_Matrix[i][4] > US_suicide_rate():
            suicide_greater_than_US = np.append(suicide_greater_than_US, World_Data_Matrix[i][1])
            weights_suicide_greater_than_US = np.append(weights_suicide_greater_than_US, World_Data_Matrix[i][5])
        i = i + 1

    weighted_mean_murder_A = calc_weighted_mean(gun_law_strictness_murder_A, weights_gun_law_strictness_A)
    weighted_std_murder_A = calc_weighted_std(gun_law_strictness_murder_A, weights_gun_law_strictness_A)
    weighted_mean_suicide_A = calc_weighted_mean(gun_law_strictness_suicide_A, weights_gun_law_strictness_A)
    weighted_std_suicide_A = calc_weighted_std(gun_law_strictness_suicide_A, weights_gun_law_strictness_A)
    weighted_mean_murder_B = calc_weighted_mean(gun_law_strictness_murder_B, weights_gun_law_strictness_B)
    weighted_std_murder_B = calc_weighted_std(gun_law_strictness_murder_B, weights_gun_law_strictness_B)
    weighted_mean_suicide_B = calc_weighted_mean(gun_law_strictness_suicide_B, weights_gun_law_strictness_B)
    weighted_std_suicide_B = calc_weighted_std(gun_law_strictness_suicide_B, weights_gun_law_strictness_B)
    weighted_mean_murder_C = calc_weighted_mean(gun_law_strictness_murder_C, weights_gun_law_strictness_C)
    weighted_std_murder_C = calc_weighted_std(gun_law_strictness_murder_C, weights_gun_law_strictness_C)
    weighted_mean_suicide_C = calc_weighted_mean(gun_law_strictness_suicide_C, weights_gun_law_strictness_C)
    weighted_std_suicide_C = calc_weighted_std(gun_law_strictness_suicide_C, weights_gun_law_strictness_C)

    weighted_mean_murder_less_than_US = calc_weighted_mean(murder_less_than_US, weights_murder_less_than_US)
    weighted_std_murder_less_than_US = calc_weighted_std(murder_less_than_US, weights_murder_less_than_US)
    weighted_mean_murder_greater_than_US = calc_weighted_mean(murder_greater_than_US, weights_murder_greater_than_US)
    weighted_std_murder_greater_than_US = calc_weighted_std(murder_greater_than_US, weights_murder_greater_than_US)
    weighted_mean_suicide_greater_than_US = calc_weighted_mean(suicide_greater_than_US, weights_suicide_greater_than_US)
    weighted_std_suicide_greater_than_US = calc_weighted_std(suicide_greater_than_US, weights_suicide_greater_than_US)
    weighted_mean_suicide_less_than_US = calc_weighted_mean(suicide_less_than_US, weights_suicide_less_than_US)
    weighted_std_suicide_less_than_US = calc_weighted_std(suicide_less_than_US, weights_suicide_less_than_US)

    t_score_murder_a_b = (weighted_mean_murder_A - weighted_mean_murder_B) / math.sqrt((math.pow(weighted_std_murder_A, 2) / len(gun_law_strictness_murder_A)) + (math.pow(weighted_std_murder_B, 2) / len(gun_law_strictness_murder_B)))
    p_value_murder_a_b = 2 * (1 - st.t.cdf(abs(t_score_murder_a_b), len(gun_law_strictness_murder_A) + len(gun_law_strictness_murder_B) - 2))
    t_score_suicide_a_b = (weighted_mean_suicide_A - weighted_mean_suicide_B) / math.sqrt((math.pow(weighted_std_suicide_A, 2) / len(gun_law_strictness_suicide_A)) + (math.pow(weighted_std_suicide_B, 2) / len(gun_law_strictness_suicide_B)))
    p_value_suicide_a_b = 2 * (1 - st.t.cdf(abs(t_score_suicide_a_b), len(gun_law_strictness_suicide_A) + len(gun_law_strictness_suicide_B) - 2))
    t_score_murder_a_c = (weighted_mean_murder_A - weighted_mean_murder_C) / math.sqrt((math.pow(weighted_std_murder_A, 2) / len(gun_law_strictness_murder_A)) + (math.pow(weighted_std_murder_C, 2) / len(gun_law_strictness_murder_C)))
    p_value_murder_a_c = 2 * (1 - st.t.cdf(abs(t_score_murder_a_c), len(gun_law_strictness_murder_A) + len(gun_law_strictness_murder_C) - 2))
    t_score_suicide_a_c = (weighted_mean_suicide_A - weighted_mean_suicide_C) / math.sqrt((math.pow(weighted_std_suicide_A, 2) / len(gun_law_strictness_suicide_A)) + (math.pow(weighted_std_suicide_C, 2) / len(gun_law_strictness_suicide_C)))
    p_value_suicide_a_c = 2 * (1 - st.t.cdf(abs(t_score_suicide_a_c), len(gun_law_strictness_suicide_A) + len(gun_law_strictness_suicide_C) - 2))
    t_score_murder_b_c = (weighted_mean_murder_B - weighted_mean_murder_C) / math.sqrt((math.pow(weighted_std_murder_B, 2) / len(gun_law_strictness_murder_B)) + (math.pow(weighted_std_murder_C, 2) / len(gun_law_strictness_murder_C)))
    p_value_murder_b_c = 2 * (1 - st.t.cdf(abs(t_score_murder_b_c), len(gun_law_strictness_murder_B) + len(gun_law_strictness_murder_C) - 2))
    t_score_suicide_b_c = (weighted_mean_suicide_B - weighted_mean_suicide_C) / math.sqrt((math.pow(weighted_std_suicide_B, 2) / len(gun_law_strictness_suicide_B)) + (math.pow(weighted_std_suicide_C, 2) / len(gun_law_strictness_suicide_C)))
    p_value_suicide_b_c = 2 * (1 - st.t.cdf(abs(t_score_suicide_b_c), len(gun_law_strictness_suicide_B) + len(gun_law_strictness_suicide_C) - 2))

    t_score_murder_US_comparison = (weighted_mean_murder_greater_than_US - weighted_mean_murder_less_than_US) / math.sqrt((math.pow(weighted_std_murder_greater_than_US, 2) / len(murder_greater_than_US)) + (math.pow(weighted_std_murder_less_than_US, 2) / len(murder_less_than_US)))
    p_value_murder_US_comparison = 2 * (1 - st.t.cdf(abs(t_score_murder_US_comparison), len(murder_less_than_US) + len(murder_greater_than_US) - 2))
    t_score_suicide_US_comparison = (weighted_mean_suicide_greater_than_US - weighted_mean_suicide_less_than_US) / math.sqrt((math.pow(weighted_std_suicide_greater_than_US, 2) / len(suicide_greater_than_US)) + (math.pow(weighted_std_suicide_less_than_US, 2) / len(suicide_less_than_US)))
    p_value_suicide_US_comparison = 2 * (1 - st.t.cdf(abs(t_score_suicide_US_comparison), len(suicide_less_than_US) + len(suicide_greater_than_US) - 2))

    print("------------------------------------------------------------T Test Outliers and Gun Laws---------------------------------------------------")
    print("The mean murder rate for the lax gun control category is " + str(weighted_mean_murder_A) + ".")
    print("The standard deviation murder rate for the lax gun control category is " + str(weighted_std_murder_A) + ".")
    print("The mean murder rate for the restrictive gun control category is " + str(weighted_mean_murder_B) + ".")
    print("The standard deviation murder rate for the restrictive gun control category is " + str(weighted_std_murder_B) + ".")
    print("The mean murder rate for the prohibited gun control category is " + str(weighted_mean_murder_C) + ".")
    print("The standard deviation murder rate for the prohibited gun control category is " + str(weighted_std_murder_C) + ".")
    print("The p value for murder between lax and restrictive gun control is " + str(p_value_murder_a_b) + ".")
    print("The p value for murder between lax and prohibited gun control is " + str(p_value_murder_a_c) + ".")
    print("The p value for murder between restrictive and prohibited gun control is " + str(p_value_murder_b_c) + ".")
    print("The mean suicide rate for the lax gun control category is " + str(weighted_mean_suicide_A) + ".")
    print("The standard deviation suicide rate for the lax gun control category is " + str(weighted_std_suicide_A) + ".")
    print("The mean suicide rate for the restrictive gun control category is " + str(weighted_mean_suicide_B) + ".")
    print("The standard deviation suicide rate for the restrictive gun control category is " + str(weighted_std_suicide_B) + ".")
    print("The mean suicide rate for the prohibited gun control category is " + str(weighted_mean_suicide_C) + ".")
    print("The standard deviation suicide rate for the prohibited gun control category is " + str(weighted_std_suicide_C) + ".")
    print("The p value for suicide between lax and restrictive gun control is " + str(p_value_suicide_a_b) + ".")
    print("The p value for suicide between lax and prohibited gun control is " + str(p_value_suicide_a_c) + ".")
    print("The p value for suicide between restrictive and prohibited gun control is " + str(p_value_suicide_b_c) + ".")

    print("------------------------------------------------------------T Test Outliers and US Rates---------------------------------------------------")
    print("The mean gun prevalence rate for places with lower murder rates than the US is " + str(weighted_mean_murder_less_than_US) + ".")
    print("The standard deviation gun prevalence rate for places with lower murder rates than the US is " + str(weighted_std_murder_less_than_US) + ".")
    print("The mean gun prevalence rate for places with higher murder rates than the US is " + str(weighted_mean_murder_greater_than_US) + ".")
    print("The standard deviation gun prevalence rate for places with higher murder rates than the US is " + str(weighted_std_murder_B) + ".")
    print("The p value for murder between lower and higher murder rates than the US is " + str(p_value_murder_US_comparison) + ".")
    print("The mean gun prevalence rate for places with lower suicide rates than the US is " + str(weighted_mean_suicide_less_than_US) + ".")
    print("The standard deviation gun prevalence rate for places with lower suicide rates than the US is " + str(weighted_std_suicide_less_than_US) + ".")
    print("The mean gun prevalence rate for places with higher suicide rates than the US is " + str(weighted_mean_suicide_greater_than_US) + ".")
    print("The standard deviation gun prevalence rate for places with higher suicide rates than the US is " + str(weighted_std_suicide_B) + ".")
    print("The p value for suicide between lower and higher suicide rates than the US is " + str(p_value_suicide_US_comparison) + ".")

    World_Data = pd.read_csv('C:/Users/dg392003/Desktop/Gun Violence Project/World Data No Outliers.csv')
    World_Data_Matrix = World_Data.values
    World_Data_Matrix = np.transpose(World_Data_Matrix)
    print(World_Data.to_string())

    gun_prevalence_rate_world = World_Data_Matrix[1]
    gun_law_strictness_world = World_Data_Matrix[2]
    murder_rate_world = World_Data_Matrix[3]
    suicide_rate_world = World_Data_Matrix[4]
    population_world = World_Data_Matrix[5]

    gun_prevalence_rate_world_mean = calc_weighted_mean(gun_prevalence_rate_world, population_world)
    gun_prevalence_rate_world_std = calc_weighted_std(gun_prevalence_rate_world, population_world)
    murder_rate_world_mean = calc_weighted_mean(murder_rate_world, population_world)
    murder_rate_world_std = calc_weighted_std(murder_rate_world, population_world)
    suicide_rate_world_mean = calc_weighted_mean(suicide_rate_world, population_world)
    suicide_rate_world_std = calc_weighted_std(suicide_rate_world, population_world)

    weighted_correlation_prevalence_murder = calc_weighted_correlation(gun_prevalence_rate_world, murder_rate_world,
                                                                       population_world)
    reg_coeff_prevalence_murder = calc_weighted_reg_coeff(gun_prevalence_rate_world, murder_rate_world,
                                                          population_world)
    slope_prevalence_murder = reg_coeff_prevalence_murder[0]
    intercept_prevalence_murder = reg_coeff_prevalence_murder[1]
    p_value_prevalence_murder = calc_p_value(0, weighted_correlation_prevalence_murder,
                                                              len(population_world))

    weighted_correlation_prevalence_suicide = calc_weighted_correlation(gun_prevalence_rate_world, suicide_rate_world,
                                                                        population_world)
    reg_coeff_prevalence_suicide = calc_weighted_reg_coeff(gun_prevalence_rate_world, suicide_rate_world,
                                                           population_world)
    slope_prevalence_suicide = reg_coeff_prevalence_suicide[0]
    intercept_prevalence_suicide = reg_coeff_prevalence_suicide[1]
    p_value_prevalence_suicide = calc_p_value(0, weighted_correlation_prevalence_suicide,
                                                               len(population_world))
    gun_prevalence_rate_world_zscore = np.zeros(len(population_world))
    gun_prevalence_rate_world_percentile = np.zeros(len(population_world))

    plot1_sizes = population_world.astype('f')  # array sizing only works for floats
    plot2_sizes = population_world.astype('f')

    i = 0
    while i < len(gun_prevalence_rate_world):
        gun_prevalence_rate_world_zscore[i] = (gun_prevalence_rate_world[
                                                   i] - gun_prevalence_rate_world_mean) / gun_prevalence_rate_world_std
        gun_prevalence_rate_world_percentile[i] = 100 * st.norm.cdf(gun_prevalence_rate_world_zscore[i])
        plot1_sizes[i] = plot1_sizes[i] / 5000000
        plot2_sizes[i] = plot2_sizes[i] / 5000000
        i = i + 1

    print(
        "------------------------------------------------------------SUMMARY STATISTICS-------------------------------------------------------")
    print("The mean world estimated gun prevalence is " + str(gun_prevalence_rate_world_mean) + ".")
    print("The standard deviation of world estimated gun prevalence is " + str(gun_prevalence_rate_world_std) + ".")
    print("The mean world murder rate per 100,000 is " + str(murder_rate_world_mean) + ".")
    print("The standard deviation of world murder rate per 100,000 is " + str(murder_rate_world_std) + ".")
    print("The mean world suicide rate per 100,000 is " + str(suicide_rate_world_mean) + ".")
    print("The standard deviation of world suicide rate per 100,000 is " + str(suicide_rate_world_std) + ".")
    print(
        "------------------------------------------------------------CORRELATION ANALYSIS-----------------------------------------------------")
    print("The weighted correlation between gun prevalence and murder rate within the world is " + str(
        weighted_correlation_prevalence_murder) + ".")
    print("This yields an R squared value of " + str(math.pow(weighted_correlation_prevalence_murder, 2)) + ".")
    print("The slope is " + str(slope_prevalence_murder) + ".")
    print("The p value is " + str(p_value_prevalence_murder) + ".")
    print("The weighted correlation between gun prevalence and suicide rate within the world is " + str(
        weighted_correlation_prevalence_suicide))
    print("This yields an R squared value of " + str(math.pow(weighted_correlation_prevalence_suicide, 2)) + ".")
    print("The slope is " + str(slope_prevalence_suicide) + ".")
    print("The p value is " + str(p_value_prevalence_suicide) + ".")

    plt.scatter(gun_prevalence_rate_world, murder_rate_world, s=plot1_sizes, c=gun_prevalence_rate_world_percentile,
                cmap='RdBu_r')
    plt.title("Murder Rate vs Gun Prevalence Rate in the World (Trimmed)")
    plt.xlabel("Estimated Gun Prevalence Rate")
    plt.ylabel("Murder Rate")
    plt.axhline(y=murder_rate_world_mean, color='k', label="Mean Murder Rate")
    plt.axvline(x=gun_prevalence_rate_world_mean, color='k', label="Mean Gun Prevalence Rate")
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Gun Prevalence Rate Percentile")
    plt.legend(prop={'size': 7})
    plt.plot(np.array([0, 60]),
             np.array([intercept_prevalence_murder, slope_prevalence_murder * 120 + intercept_prevalence_murder]),
             color='k')
    plt.show()

    plt.scatter(gun_prevalence_rate_world, suicide_rate_world, s=plot1_sizes, c=gun_prevalence_rate_world_percentile,
                cmap='RdBu_r')
    plt.title("Suicide Rate vs Gun Prevalence Rate in the World (Trimmed)")
    plt.xlabel("Estimated Gun Prevalence Rate")
    plt.ylabel("Suicide Rate")
    plt.axhline(y=suicide_rate_world_mean, color='k', label="Mean Suicide Rate")
    plt.axvline(x=gun_prevalence_rate_world_mean, color='k', label="Mean Gun Ownership Rate")
    cbar2 = plt.colorbar()
    cbar2.ax.set_ylabel("Gun Prevalence Rate Percentile")
    plt.legend(prop={'size': 7})
    plt.plot(np.array([0, 60]),
             np.array([intercept_prevalence_suicide, slope_prevalence_suicide * 120 + intercept_prevalence_suicide]),
             color='k')
    plt.show()

    World_Data_Matrix = np.transpose(World_Data_Matrix)
    gun_law_strictness_murder_A = np.array([])
    gun_law_strictness_suicide_A = np.array([])
    weights_gun_law_strictness_A = np.array([])
    gun_law_strictness_murder_B = np.array([])
    gun_law_strictness_suicide_B = np.array([])
    weights_gun_law_strictness_B = np.array([])
    gun_law_strictness_murder_C = np.array([])
    gun_law_strictness_suicide_C = np.array([])
    weights_gun_law_strictness_C = np.array([])

    murder_less_than_US = np.array([])
    weights_murder_less_than_US = np.array([])
    murder_greater_than_US = np.array([])
    weights_murder_greater_than_US = np.array([])
    suicide_less_than_US = np.array([])
    weights_suicide_less_than_US = np.array([])
    suicide_greater_than_US = np.array([])
    weights_suicide_greater_than_US = np.array([])

    i = 0
    while i < len(population_world):
        if World_Data_Matrix[i][2] == 'A':
            gun_law_strictness_murder_A = np.append(gun_law_strictness_murder_A, World_Data_Matrix[i][3])
            gun_law_strictness_suicide_A = np.append(gun_law_strictness_suicide_A, World_Data_Matrix[i][4])
            weights_gun_law_strictness_A = np.append(weights_gun_law_strictness_A, World_Data_Matrix[i][5])
        elif World_Data_Matrix[i][2] == 'B':
            gun_law_strictness_murder_B = np.append(gun_law_strictness_murder_B, World_Data_Matrix[i][3])
            gun_law_strictness_suicide_B = np.append(gun_law_strictness_suicide_B, World_Data_Matrix[i][4])
            weights_gun_law_strictness_B = np.append(weights_gun_law_strictness_B, World_Data_Matrix[i][5])
        elif World_Data_Matrix[i][2] == 'C':
            gun_law_strictness_murder_C = np.append(gun_law_strictness_murder_C, World_Data_Matrix[i][3])
            gun_law_strictness_suicide_C = np.append(gun_law_strictness_suicide_C, World_Data_Matrix[i][4])
            weights_gun_law_strictness_C = np.append(weights_gun_law_strictness_C, World_Data_Matrix[i][5])
        if World_Data_Matrix[i][3] < US_murder_rate():
            murder_less_than_US = np.append(murder_less_than_US, World_Data_Matrix[i][1])
            weights_murder_less_than_US = np.append(weights_murder_less_than_US, World_Data_Matrix[i][5])
        elif World_Data_Matrix[i][3] > US_murder_rate():
            murder_greater_than_US = np.append(murder_greater_than_US, World_Data_Matrix[i][1])
            weights_murder_greater_than_US = np.append(weights_murder_greater_than_US, World_Data_Matrix[i][5])
        if World_Data_Matrix[i][4] < US_suicide_rate():
            suicide_less_than_US = np.append(suicide_less_than_US, World_Data_Matrix[i][1])
            weights_suicide_less_than_US = np.append(weights_suicide_less_than_US, World_Data_Matrix[i][5])
        elif World_Data_Matrix[i][4] > US_suicide_rate():
            suicide_greater_than_US = np.append(suicide_greater_than_US, World_Data_Matrix[i][1])
            weights_suicide_greater_than_US = np.append(weights_suicide_greater_than_US, World_Data_Matrix[i][5])
        i = i + 1

    weighted_mean_murder_A = calc_weighted_mean(gun_law_strictness_murder_A, weights_gun_law_strictness_A)
    weighted_std_murder_A = calc_weighted_std(gun_law_strictness_murder_A, weights_gun_law_strictness_A)
    weighted_mean_suicide_A = calc_weighted_mean(gun_law_strictness_suicide_A, weights_gun_law_strictness_A)
    weighted_std_suicide_A = calc_weighted_std(gun_law_strictness_suicide_A, weights_gun_law_strictness_A)
    weighted_mean_murder_B = calc_weighted_mean(gun_law_strictness_murder_B, weights_gun_law_strictness_B)
    weighted_std_murder_B = calc_weighted_std(gun_law_strictness_murder_B, weights_gun_law_strictness_B)
    weighted_mean_suicide_B = calc_weighted_mean(gun_law_strictness_suicide_B, weights_gun_law_strictness_B)
    weighted_std_suicide_B = calc_weighted_std(gun_law_strictness_suicide_B, weights_gun_law_strictness_B)
    weighted_mean_murder_C = calc_weighted_mean(gun_law_strictness_murder_C, weights_gun_law_strictness_C)
    weighted_std_murder_C = calc_weighted_std(gun_law_strictness_murder_C, weights_gun_law_strictness_C)
    weighted_mean_suicide_C = calc_weighted_mean(gun_law_strictness_suicide_C, weights_gun_law_strictness_C)
    weighted_std_suicide_C = calc_weighted_std(gun_law_strictness_suicide_C, weights_gun_law_strictness_C)

    weighted_mean_murder_less_than_US = calc_weighted_mean(murder_less_than_US, weights_murder_less_than_US)
    weighted_std_murder_less_than_US = calc_weighted_std(murder_less_than_US, weights_murder_less_than_US)
    weighted_mean_murder_greater_than_US = calc_weighted_mean(murder_greater_than_US, weights_murder_greater_than_US)
    weighted_std_murder_greater_than_US = calc_weighted_std(murder_greater_than_US, weights_murder_greater_than_US)
    weighted_mean_suicide_greater_than_US = calc_weighted_mean(suicide_greater_than_US, weights_suicide_greater_than_US)
    weighted_std_suicide_greater_than_US = calc_weighted_std(suicide_greater_than_US, weights_suicide_greater_than_US)
    weighted_mean_suicide_less_than_US = calc_weighted_mean(suicide_less_than_US, weights_suicide_less_than_US)
    weighted_std_suicide_less_than_US = calc_weighted_std(suicide_less_than_US, weights_suicide_less_than_US)

    t_score_murder_a_b = (weighted_mean_murder_A - weighted_mean_murder_B) / math.sqrt(
        (math.pow(weighted_std_murder_A, 2) / len(gun_law_strictness_murder_A)) + (
                    math.pow(weighted_std_murder_B, 2) / len(gun_law_strictness_murder_B)))
    p_value_murder_a_b = 2 * (1 - st.t.cdf(abs(t_score_murder_a_b),
                                           len(gun_law_strictness_murder_A) + len(gun_law_strictness_murder_B) - 2))
    t_score_suicide_a_b = (weighted_mean_suicide_A - weighted_mean_suicide_B) / math.sqrt(
        (math.pow(weighted_std_suicide_A, 2) / len(gun_law_strictness_suicide_A)) + (
                    math.pow(weighted_std_suicide_B, 2) / len(gun_law_strictness_suicide_B)))
    p_value_suicide_a_b = 2 * (1 - st.t.cdf(abs(t_score_suicide_a_b),
                                            len(gun_law_strictness_suicide_A) + len(gun_law_strictness_suicide_B) - 2))
    t_score_murder_a_c = (weighted_mean_murder_A - weighted_mean_murder_C) / math.sqrt(
        (math.pow(weighted_std_murder_C, 2) / len(gun_law_strictness_murder_A)) + (
                    math.pow(weighted_std_murder_C, 2) / len(gun_law_strictness_murder_C)))
    p_value_murder_a_c = 2 * (1 - st.t.cdf(abs(t_score_murder_a_c),
                                           len(gun_law_strictness_murder_C) + len(gun_law_strictness_murder_C) - 2))
    t_score_suicide_a_c = (weighted_mean_suicide_A - weighted_mean_suicide_C) / math.sqrt(
        (math.pow(weighted_std_suicide_C, 2) / len(gun_law_strictness_suicide_A)) + (
                    math.pow(weighted_std_suicide_C, 2) / len(gun_law_strictness_suicide_C)))
    p_value_suicide_a_c = 2 * (1 - st.t.cdf(abs(t_score_suicide_a_c),
                                            len(gun_law_strictness_suicide_C) + len(gun_law_strictness_suicide_C) - 2))
    t_score_murder_b_c = (weighted_mean_murder_B - weighted_mean_murder_C) / math.sqrt(
        (math.pow(weighted_std_murder_B, 2) / len(gun_law_strictness_murder_B)) + (
                    math.pow(weighted_std_murder_C, 2) / len(gun_law_strictness_murder_C)))
    p_value_murder_b_c = 2 * (1 - st.t.cdf(abs(t_score_murder_b_c),
                                           len(gun_law_strictness_murder_B) + len(gun_law_strictness_murder_C) - 2))
    t_score_suicide_b_c = (weighted_mean_suicide_B - weighted_mean_suicide_C) / math.sqrt(
        (math.pow(weighted_std_suicide_B, 2) / len(gun_law_strictness_suicide_C)) + (
                    math.pow(weighted_std_suicide_C, 2) / len(gun_law_strictness_suicide_C)))
    p_value_suicide_b_c = 2 * (1 - st.t.cdf(abs(t_score_suicide_b_c),
                                            len(gun_law_strictness_suicide_B) + len(gun_law_strictness_suicide_C) - 2))

    t_score_murder_US_comparison = (
                                               weighted_mean_murder_greater_than_US - weighted_mean_murder_less_than_US) / math.sqrt(
        (math.pow(weighted_std_murder_greater_than_US, 2) / len(murder_greater_than_US)) + (
                    math.pow(weighted_std_murder_less_than_US, 2) / len(murder_less_than_US)))
    p_value_murder_US_comparison = 2 * (1 - st.t.cdf(abs(t_score_murder_US_comparison),
                                                     len(murder_less_than_US) + len(murder_greater_than_US) - 2))
    t_score_suicide_US_comparison = (
                                                weighted_mean_suicide_greater_than_US - weighted_mean_suicide_less_than_US) / math.sqrt(
        (math.pow(weighted_std_suicide_greater_than_US, 2) / len(suicide_greater_than_US)) + (
                    math.pow(weighted_std_suicide_less_than_US, 2) / len(suicide_less_than_US)))
    p_value_suicide_US_comparison = 2 * (1 - st.t.cdf(abs(t_score_suicide_US_comparison),
                                                      len(suicide_less_than_US) + len(suicide_greater_than_US) - 2))

    print(
        "------------------------------------------------------------T Test No Outliers and Gun Laws---------------------------------------------------")
    print("The mean murder rate for the lax gun control category is " + str(weighted_mean_murder_A) + ".")
    print("The standard deviation murder rate for the lax gun control category is " + str(weighted_std_murder_A) + ".")
    print("The mean murder rate for the restrictive gun control category is " + str(weighted_mean_murder_B) + ".")
    print("The standard deviation murder rate for the restrictive gun control category is " + str(
        weighted_std_murder_B) + ".")
    print("The mean murder rate for the prohibited gun control category is " + str(weighted_mean_murder_C) + ".")
    print("The standard deviation murder rate for the prohibited gun control category is " + str(
        weighted_std_murder_C) + ".")
    print("The p value for murder between lax and restrictive gun control is " + str(p_value_murder_a_b) + ".")
    print("The p value for murder between lax and prohibited gun control is " + str(p_value_murder_a_c) + ".")
    print("The p value for murder between restrictive and prohibited gun control is " + str(p_value_murder_b_c) + ".")
    print("The mean suicide rate for the lax gun control category is " + str(weighted_mean_suicide_A) + ".")
    print(
        "The standard deviation suicide rate for the lax gun control category is " + str(weighted_std_suicide_A) + ".")
    print("The mean suicide rate for the restrictive gun control category is " + str(weighted_mean_suicide_B) + ".")
    print("The standard deviation suicide rate for the restrictive gun control category is " + str(
        weighted_std_suicide_B) + ".")
    print("The mean suicide rate for the prohibited gun control category is " + str(weighted_mean_suicide_C) + ".")
    print("The standard deviation suicide rate for the prohibited gun control category is " + str(
        weighted_std_suicide_C) + ".")
    print("The p value for suicide between lax and restrictive gun control is " + str(p_value_suicide_a_b) + ".")
    print("The p value for suicide between lax and prohibited gun control is " + str(p_value_suicide_a_c) + ".")
    print("The p value for suicide between restrictive and prohibited gun control is " + str(p_value_suicide_b_c) + ".")

    print(
        "------------------------------------------------------------T Test No Outliers and US Rates---------------------------------------------------")
    print("The mean gun prevalence rate for places with lower murder rates than the US is " + str(
        weighted_mean_murder_less_than_US) + ".")
    print("The standard deviation gun prevalence rate for places with lower murder rates than the US is " + str(
        weighted_std_murder_less_than_US) + ".")
    print("The mean gun prevalence rate for places with higher murder rates than the US is " + str(
        weighted_mean_murder_greater_than_US) + ".")
    print("The standard deviation gun prevalence rate for places with higher murder rates than the US is " + str(
        weighted_std_murder_B) + ".")
    print("The p value for murder between lower and higher murder rates than the US is " + str(
        p_value_murder_US_comparison) + ".")
    print("The mean gun prevalence rate for places with lower suicide rates than the US is " + str(
        weighted_mean_suicide_less_than_US) + ".")
    print("The standard deviation gun prevalence rate for places with lower suicide rates than the US is " + str(
        weighted_std_suicide_less_than_US) + ".")
    print("The mean gun prevalence rate for places with higher suicide rates than the US is " + str(
        weighted_mean_suicide_greater_than_US) + ".")
    print("The standard deviation gun prevalence rate for places with higher suicide rates than the US is " + str(
        weighted_std_suicide_B) + ".")
    print("The p value for suicide between lower and higher suicide rates than the US is " + str(
        p_value_suicide_US_comparison) + ".")

US_output()
DevelopedCountry_output()
World_output()



