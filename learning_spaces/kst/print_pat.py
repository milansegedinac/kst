
def print_pat(x):
    """
    Formatted print of pattern response
    :param x: dictionary - response from pattern function
    :return:
    """
    print('\nlargest response patterns in the data: {}'.format(x['n']))
    print(x['response.patterns'])
    if(x['states'] != None):
        print("Number of times a state occurs in the data:")
        print(x['states'])
