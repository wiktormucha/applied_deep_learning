'''
File containing most basic functions 
'''


def count_parameters(model):
    '''
    Function to count parameters in a model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)