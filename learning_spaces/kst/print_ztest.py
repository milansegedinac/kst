import numpy as np

def print_ztest(x):
    if len(x['diff_value']) == 1:
        print("\n One sample Z-test\n")
    if len(x['diff_value']) == 2:
        print("\n \t Two sample Z-test\n")
    print("\nz = {}".format(round(x['Z.value'], 4)))
    print("p-value = {}".format(round(x['p.value'], 4)))
    if "two.sided" == x['alternative']:
        if x is None:
            print("\nalternative hypothesis: true mean is not equal {}".format(x['mu']))
        else:
            print("\nalternative hypothesis: true difference in means is not equal {}".format(x['mu']))

    if "greater" == x['alternative']:
        if x['imp_alt'] is None:
            print("\nalternative hypothesis: true mean is greater {}".format(x['mu']))
        else:
            print("\nalternative hypothesis: true difference in means is greater {}".format(x['mu']))

    if x['alternative'] == "less":
        if x['imp_alt'] is None:
            print("\nalternative hypothesis: true mean is less {}".format(x['mu']))
        else:
            print("\nalternative hypothesis: true difference in means is less {}".format(x['mu']))

    print(str(x['conf.level'] * 100) + " percent confidence interval:\n")
    print(x['conf'])
    print("sample estimates:\n")
    if len(x['diff_value']) == 1:
        estimate = round(x['diff_value'][0],  5)
        names = {}
        names[estimate] = "mean in imp"
        print(estimate)
    if len(x['diff_value']) == 2:
        estimate = round(x['diff_value'], 5)
        names = {}
        names[estimate] = "mean in imp_alt"
        print(estimate)
