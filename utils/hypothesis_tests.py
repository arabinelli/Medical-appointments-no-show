from scipy.stats import ttest_ind
from typing import List
Vector = List[int]


def hypothesis_test_binary_distribution(a: Vector,b: Vector,column_name: str,
                                        values: Vector) -> None:
    """
    Takes two distributions and perform a two-sided hypothesis tests to assess if
    the distributions are statistically different. The test is performed at 95% and 90% 
    significance.
    
    ARGUMENTS:
    a (iterable, array-like) - first distribution to compare
    b (iterable, array-like) - second distribution to compare
    column_name (str) - the name of the column for which the hypothesis test is being run
    values (str) - the values that each distribution takes (e.g. "males" and "females")

    """
    p_value = ttest_ind(a=a,
                        b=b).pvalue
    
    base_string = "The distributions of no-show between " + column_name + " = " + str(values[0]) + " and "\
                  + column_name + " = " + str(values[1])
    if p_value < 0.05:
        print("\033[94m")
        print(base_string + " are statistically different with a significance of 95%" )
        print("p-value:",p_value)
        print("\x1b[0m")
    elif p_value < 0.1:
        print("\033[93m")
        print(base_string + " are statistically different with a significance of 90%" )
        print("p-value:",p_value)
        print("\x1b[0m")
    else:
        print("\x1b[31m")
        print(base_string + " are not statistically different")
        print("p-value:",p_value)
        print("\x1b[0m")