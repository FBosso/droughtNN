#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers for building and parsing the '%'-separated variable-path "combo"
strings used to describe a feature combination across the daily_nn pipeline.
"""


def combo2pretty(combo):
    '''
    Parameters
    ----------
    combo : TYPE str
        DESCRIPTION. combo string composed by the path of all the variables
        separated by '%' sign

    Returns a string of all the involved variables separated by '-'
    -------
    None.
    '''

    paths = combo.split('%')
    variables = ''
    for i, path in enumerate(paths):
        variable = path.split('/')[-1]
        if i != 0:
            variables = variables + '-' + variable
        elif i == 0:
            variables = variables + variable

    return variables


def tuple2combostring(tup):
    '''
    Parameters
    ----------
    tup : TYPE tuple
        DESCRIPTION. tuple containing the paths of all the variable of a single combination

    Returns the generating string to give as input to the generate_full_dataset function
    -------
    None.
    '''
    string = ''
    for i, item in enumerate(tup):
        if i == 0:
            string = string + item
        else:
            string = string + '%' + item

    return string


def detect_global(combo):

    gl_vars = ['SST', 'MSLP', 'Z500']

    paths = combo.split('%')
    global_paths = []
    local_paths = []
    for path in paths:
        if path.split('/')[-1] in gl_vars:
            global_paths.append(path)
        else:
            local_paths.append(path)

    local_string = ''
    for i, path in enumerate(local_paths):
        if i == 0:
            local_string = local_string + path
        else:
            local_string = local_string + '%' + path

    return global_paths, local_string


def gen2gens(gen_string):
    '''
    Parameters
    ----------
    gen_string : TYPE str
        DESCRIPTION. string containing the paths of all the variables (global and local)

    Returns two generating strings containing the paths of the data separated by %,
    one for the local vars and the other for the global one
    -------
    None.
    '''

    glob_list = ['MSLP', 'SST', 'Z500']
    loc_list = ['MER', 'MSSHF', 'RH', 'SD', 'SH', 't2m', 'TCC', 'TCWV', 'tp', 'UW', 'VW']

    paths = gen_string.split('%')
    glob = ''
    loc = ''
    for path in paths:
        if path.split('/')[-1] in glob_list:
            if len(glob) == 0:
                glob = glob + path
            else:
                glob = glob + '%' + path
        elif path.split('/')[-1] in loc_list:
            if len(loc) == 0:
                loc = loc + path
            else:
                loc = loc + '%' + path

    return loc, glob


def gen_signals(gen_string):
    '''
    Parameters
    ----------
    gen_string : TYPE str
        DESCRIPTION. string containing the paths of all the variables (global and local)

    Returns a generating scring to load the data related to climate signals (e.g NAO, ENSO, etc.)
    -------
    None.
    '''

    sig_list = ['NAO', 'SCA', 'EA', 'ENSO']

    paths = gen_string.split('%')
    sig = ''
    for path in paths:
        if path.split('/')[-1] in sig_list:
            if len(sig) == 0:
                sig = sig + path
            else:
                sig = sig + '%' + path

    return sig
